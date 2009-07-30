#include <iomanip>
#include <ostream>
#include <sstream>
#include <cstring>

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"

namespace sistrip {

  FEDBuffer::FEDBuffer(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowBadBuffer)
    : FEDBufferBase(fedBuffer,fedBufferSize,allowBadBuffer,false)
  {
    channels_.reserve(FEDCH_PER_FED);
    //build the correct type of FE header object
    if (headerType() != HEADER_TYPE_INVALID) {
      feHeader_ = FEDFEHeader::newFEHeader(headerType(),getPointerToDataAfterTrackerSpecialHeader());
      payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader()+feHeader_->lengthInBytes();
    } else {
      feHeader_ = std::auto_ptr<FEDFEHeader>();
      payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader();
      if (!allowBadBuffer) {
	std::ostringstream ss;
	ss << "Header type is invalid. "
	   << "Header type nibble is ";
	uint8_t headerNibble = trackerSpecialHeader().headerTypeNibble();
	printHex(&headerNibble,1,ss);
	ss << ". ";
	throw cms::Exception("FEDBuffer") << ss.str();
      }
    }
    payloadLength_ = getPointerToByteAfterEndOfPayload()-payloadPointer_;
    //check if FE units are present in data
    //in Full Debug mode, use the lengths from the header
    const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
    if (fdHeader) {
      for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
        if (fdHeader->fePresent(iFE)) fePresent_[iFE] = true;
        else fePresent_[iFE] = false;
      }
    }
    //in APV error mode, use the FE present byte in the FED status register
    // a value of '1' means a FE unit's data is missing (in old firmware versions it is always 0)
    else {
      for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (fedStatusRegister().feDataMissingFlag(iFE)) fePresent_[iFE] = false;
      else fePresent_[iFE] = true;
      }
    }
    //try to find channels
    validChannels_ = 0;
    try {
      findChannels();
    } catch (const cms::Exception& e) {
      //if there was a problem either rethrow the exception or just mark channel pointers NULL
      if (!allowBadBuffer) throw;
      else {
        channels_.insert(channels_.end(),size_t(FEDCH_PER_FED-validChannels_),FEDChannel(payloadPointer_,0));
      }
    }
  }

  FEDBuffer::~FEDBuffer()
  {
  }

  void FEDBuffer::findChannels()
  {
    size_t offsetBeginningOfChannel = 0;
    for (size_t i = 0; i < FEDCH_PER_FED; i++) {
      //if FE unit is not enabled then skip rest of FE unit adding NULL pointers
      if (!feGood(i/FEDCH_PER_FEUNIT)) {
	channels_.insert(channels_.end(),size_t(FEDCH_PER_FEUNIT),FEDChannel(payloadPointer_,0));
	i += FEDCH_PER_FEUNIT-1;
	validChannels_ += FEDCH_PER_FEUNIT;
	continue;
      }
      //if FE unit is enabled
      //check that channel length bytes fit into buffer
      if (offsetBeginningOfChannel+1 >= payloadLength_) {
	std::ostringstream ss;
        SiStripFedKey key(0,i/FEDCH_PER_FEUNIT,i%FEDCH_PER_FEUNIT);
        ss << "Channel " << uint16_t(i) << " (FE unit " << key.feUnit() << " channel " << key.feChan() << " according to external numbering scheme)" 
           << " does not fit into buffer. "
           << "Channel starts at " << uint16_t(offsetBeginningOfChannel) << " in payload. "
           << "Payload length is " << uint16_t(payloadLength_) << ". ";
        throw cms::Exception("FEDBuffer") << ss.str();
      }
      channels_.push_back(FEDChannel(payloadPointer_,offsetBeginningOfChannel));
      //get length and check that whole channel fits into buffer
      uint16_t channelLength = channels_.back().length();
      if (offsetBeginningOfChannel+channelLength > payloadLength_) {
        SiStripFedKey key(0,i/FEDCH_PER_FEUNIT,i%FEDCH_PER_FEUNIT);
	std::ostringstream ss;
        ss << "Channel " << uint16_t(i) << " (FE unit " << key.feUnit() << " channel " << key.feChan() << " according to external numbering scheme)" 
           << "Channel starts at " << uint16_t(offsetBeginningOfChannel) << " in payload. "
           << "Channel length is " << uint16_t(channelLength) << ". "
           << "Payload length is " << uint16_t(payloadLength_) << ". ";
        throw cms::Exception("FEDBuffer") << ss.str();
      }
      validChannels_++;
      const size_t offsetEndOfChannel = offsetBeginningOfChannel+channelLength;
      //add padding if necessary and calculate offset for begining of next channel
      if (!( (i+1) % FEDCH_PER_FEUNIT )) {
	uint8_t numPaddingBytes = 8 - (offsetEndOfChannel % 8);
	if (numPaddingBytes == 8) numPaddingBytes = 0;
	offsetBeginningOfChannel = offsetEndOfChannel + numPaddingBytes;
      } else {
	offsetBeginningOfChannel = offsetEndOfChannel;
      }
    }
  }
  
  bool FEDBuffer::channelGood(const uint8_t internalFEDChannelNum) const
  {
    return ( (internalFEDChannelNum < validChannels_) &&
             feGood(internalFEDChannelNum/FEDCH_PER_FEUNIT) &&
             checkStatusBits(internalFEDChannelNum) );
  }

  bool FEDBuffer::doChecks() const
  {
    //check that all channels were unpacked properly
    if (validChannels_ != FEDCH_PER_FED) return false;
    //do checks from base class
    if (!FEDBufferBase::doChecks()) return false;
    return true;
  }

  bool FEDBuffer::doCorruptBufferChecks() const
  {
    return ( checkCRC() &&
	     checkChannelLengthsMatchBufferLength() &&
	     checkChannelPacketCodes() &&
	     //checkClusterLengths() &&
	     checkFEUnitLengths() );
    //checkFEUnitAPVAddresses() );
  }

  bool FEDBuffer::checkAllChannelStatusBits() const
  {
    for (uint8_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
      //if FE unit is disabled then skip all channels on it
      if (!feGood(iCh/FEDCH_PER_FEUNIT)) {
	iCh += FEDCH_PER_FEUNIT;
	continue;
      }
      //channel is bad then return false
      if (!checkStatusBits(iCh)) return false;
    }
    //if no bad channels have been found then they are all fine
    return true;
  }

  bool FEDBuffer::checkChannelLengths() const
  {
    return (validChannels_ == FEDCH_PER_FED);
  }

  bool FEDBuffer::checkChannelLengthsMatchBufferLength() const
  {
    //check they fit into buffer
    if (!checkChannelLengths()) return false;
  
    //payload length from length of data buffer
    const size_t payloadLengthInWords = payloadLength_/8;
  
    //find channel length
    //find last enabled FE unit
    uint8_t lastEnabledFeUnit = 7;
    while (!feGood(lastEnabledFeUnit)) lastEnabledFeUnit--;
    //last channel is last channel on last enabled FE unit
    const FEDChannel& lastChannel = channels_[internalFEDChannelNum(lastEnabledFeUnit,FEDCH_PER_FEUNIT-1)];
    const size_t offsetLastChannel = lastChannel.offset();
    const size_t offsetEndOfChannelData = offsetLastChannel+lastChannel.length();
    const size_t channelDataLength = offsetEndOfChannelData;
    //channel length in words is length in bytes rounded up to nearest word
    size_t channelDataLengthInWords = channelDataLength/8;
    if (channelDataLength % 8) channelDataLengthInWords++;
  
    //check lengths match
    if (channelDataLengthInWords == payloadLengthInWords) {
      return true;
    } else {
      return false;
    }
  }

  bool FEDBuffer::checkChannelPacketCodes() const
  {
    const uint8_t correctPacketCode = getCorrectPacketCode();
    //if the readout mode if not one which has a packet code then this is set to zero. in this case return true
    if (!correctPacketCode) return true;
    for (uint8_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
      //if FE unit is disabled then skip all channels on it
      if (!feGood(iCh/FEDCH_PER_FEUNIT)) {
	iCh += FEDCH_PER_FEUNIT;
	continue;
      }
      //only check enabled, working channels
      if (channelGood(iCh)) {
	//if a channel is bad then return false
	if (channels_[iCh].packetCode() != correctPacketCode) return false;
      }
    }
    //if no bad channels were found the they are all ok
    return true;
  }

  bool FEDBuffer::checkFEUnitAPVAddresses() const
  {
    //check can only be done for full debug headers
    const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
    if (!fdHeader) return true;
    //get golden address
    const uint8_t goldenAddress = apveAddress();
    //check all enabled FE units
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!feGood(iFE)) continue;
      //if address is bad then return false
      if (fdHeader->feUnitMajorityAddress(iFE) != goldenAddress) return false;
    }
    //if no bad addresses were found then return true
    return true;
  }

  bool FEDBuffer::checkFEUnitLengths() const
  {
    //check can only be done for full debug headers
    const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
    if (!fdHeader) return true;
    //check lengths for enabled FE units
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!feGood(iFE)) continue;
      if (calculateFEUnitLength(iFE) != fdHeader->feUnitLength(iFE)) return false;
    }
    //if no errors were encountered then return true
    return true;
  }
  
  uint16_t FEDBuffer::calculateFEUnitLength(const uint8_t internalFEUnitNumber) const
  {
    //get length from channels
    uint16_t lengthFromChannels = 0;
    for (uint8_t iCh = 0; iCh < FEDCH_PER_FEUNIT; iCh++) {
      lengthFromChannels += channels_[internalFEDChannelNum(internalFEUnitNumber,iCh)].length();
    }
    return lengthFromChannels;
  }
  
  bool FEDBuffer::checkFEPayloadsPresent() const
  {
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!fePresent(iFE)) return false;
    }
    return true;
  }

  std::string FEDBuffer::checkSummary() const
  {
    std::stringstream summary;
    summary << FEDBufferBase::checkSummary();
    summary << "Check FE unit payloads are all present: " << (checkFEPayloadsPresent() ? "passed" : "FAILED" ) << std::endl;
    if (!checkFEPayloadsPresent()) {
      summary << "FE units missing payloads: ";
      for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
        if (!fePresent(iFE)) summary << uint16_t(iFE) << " ";
      }
      summary << std::endl;
    }
    summary << "Check channel status bits: " << ( checkAllChannelStatusBits() ? "passed" : "FAILED" ) << std::endl;
    if (!checkAllChannelStatusBits()) {
      unsigned int badChannels = 0;
      if (headerType() == HEADER_TYPE_FULL_DEBUG) {
        const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
        if (fdHeader) {
          for (uint8_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
            if (!feGood(iCh/FEDCH_PER_FEUNIT)) continue;
            if (!checkStatusBits(iCh)) {
              summary << uint16_t(iCh) << ": " << fdHeader->getChannelStatus(iCh) << std::endl;
              badChannels++;
            }
          }
        }
      } else {
        summary << "Channels with errors: ";
        for (uint8_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
          if (!feGood(iCh/FEDCH_PER_FEUNIT)) continue;
          if (!checkStatusBits(iCh)) {
            summary << uint16_t(iCh) << " ";
            badChannels++;
          }
        }
        summary << std::endl;
      } 
      summary << "Number of channels with bad status bits: " << badChannels << std::endl;
    }
    summary << "Check channel lengths match buffer length: " << ( checkChannelLengthsMatchBufferLength() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check channel packet codes: " << ( checkChannelPacketCodes() ? "passed" : "FAILED" ) << std::endl;
    if (!checkChannelPacketCodes()) {
      summary << "Channels with bad packet codes: ";
      for (uint8_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
        if (!feGood(iCh/FEDCH_PER_FEUNIT)) continue;
        if (channels_[iCh].packetCode() != getCorrectPacketCode())
          summary << uint16_t(iCh) << " ";
      }
    }
    summary << "Check FE unit lengths: " << ( checkFEUnitLengths() ? "passed" : "FAILED" ) << std::endl;
    if (!checkFEUnitLengths()) {
      const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
      if (fdHeader) {
        summary << "Bad FE units:" << std::endl;
        for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
          if (!feGood(iFE)) continue;
          uint16_t lengthFromChannels = calculateFEUnitLength(iFE);
          uint16_t lengthFromHeader = fdHeader->feUnitLength(iFE);
          if (lengthFromHeader != lengthFromChannels) {
            summary << "FE unit: " << uint16_t(iFE) 
                    << " length in header: " << lengthFromHeader 
                    << " length from channel lengths: " << lengthFromChannels << std::endl;
          }
        }
      }
    }
    summary << "Check FE unit APV addresses match APVe: " << ( checkFEUnitAPVAddresses() ? "passed" : "FAILED" ) << std::endl;
    if (!checkFEUnitAPVAddresses()) {
      const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
      if (fdHeader) {
        const uint8_t goldenAddress = apveAddress();
        summary << "Address from APVe:" << uint16_t(goldenAddress) << std::endl;
        summary << "Bad FE units:" << std::endl;
        for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
          if (!feGood(iFE)) continue;
          if (fdHeader->feUnitMajorityAddress(iFE) != goldenAddress) {
            summary << "FE unit: " << uint16_t(iFE)
                    << " majority address: " << uint16_t(fdHeader->feUnitMajorityAddress(iFE)) << std::endl;
          }
        }
      }
    }
    return summary.str();
  }

  uint8_t FEDBuffer::getCorrectPacketCode() const
  {
    switch(readoutMode()) {
    case READOUT_MODE_SCOPE:
      return PACKET_CODE_SCOPE;
      break;
    case READOUT_MODE_VIRGIN_RAW:
      return PACKET_CODE_VIRGIN_RAW;
      break;
    case READOUT_MODE_PROC_RAW:
      return PACKET_CODE_PROC_RAW;
      break;
    case READOUT_MODE_ZERO_SUPPRESSED:
      return PACKET_CODE_ZERO_SUPPRESSED;
      break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE:
    case READOUT_MODE_INVALID:
    default:
      return 0;
    }
  }

  uint8_t FEDBuffer::nFEUnitsPresent() const
  {
    uint8_t result = 0;
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (fePresent(iFE)) result++;
    }
    return result;
  }
  
  void FEDBuffer::print(std::ostream& os) const
  {
    FEDBufferBase::print(os);
    if (headerType() == HEADER_TYPE_FULL_DEBUG) {
      os << "FE units with data: " << uint16_t(nFEUnitsPresent()) << std::endl;
      os << "BE status register flags: ";
      dynamic_cast<const FEDFullDebugHeader*>(feHeader())->beStatusRegister().printFlags(os);
      os << std::endl;
    }
  }




  FEDBufferBase::FEDBufferBase(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowUnrecognizedFormat)
    : channels_(FEDCH_PER_FED,FEDChannel(NULL,0,0)),
      originalBuffer_(fedBuffer),
      bufferSize_(fedBufferSize)
  {
    init(fedBuffer,fedBufferSize,allowUnrecognizedFormat);
  }
  
  FEDBufferBase::FEDBufferBase(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowUnrecognizedFormat, const bool fillChannelVector)
    : originalBuffer_(fedBuffer),
      bufferSize_(fedBufferSize)
  {
    init(fedBuffer,fedBufferSize,allowUnrecognizedFormat);
    if (fillChannelVector) channels_.assign(FEDCH_PER_FED,FEDChannel(NULL,0,0));
  }
  
  void FEDBufferBase::init(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowUnrecognizedFormat)
  {
    //min buffer length. DAQ header, DAQ trailer, tracker special header. 
    static const size_t MIN_BUFFER_SIZE = 8+8+8;
    //check size is non zero and data pointer is not NULL
    if (!originalBuffer_) throw cms::Exception("FEDBuffer") << "Buffer pointer is NULL.";
    if (bufferSize_ < MIN_BUFFER_SIZE) {
      std::ostringstream ss;
      ss << "Buffer is too small. "
         << "Min size is " << MIN_BUFFER_SIZE << ". "
         << "Buffer size is " << bufferSize_ << ". ";
      throw cms::Exception("FEDBuffer") << ss.str();
    }
  
    //construct tracker special header using second 64 bit word
    specialHeader_ = TrackerSpecialHeader(originalBuffer_+8);
  
    //check the buffer format
    const FEDBufferFormat bufferFormat = specialHeader_.bufferFormat();
    if (bufferFormat == BUFFER_FORMAT_INVALID && !allowUnrecognizedFormat) {
      std::ostringstream ss;
      ss << "Buffer format not recognized. "
         << "Tracker special header: " << specialHeader_;
      throw cms::Exception("FEDBuffer") << ss.str();
    }
    //swap the buffer words so that the whole buffer is in slink ordering
    if ( (bufferFormat == BUFFER_FORMAT_OLD_VME) || (bufferFormat == BUFFER_FORMAT_NEW) ) {
      uint8_t* newBuffer = new uint8_t[bufferSize_];
      const uint32_t* originalU32 = reinterpret_cast<const uint32_t*>(originalBuffer_);
      const size_t sizeU32 = bufferSize_/4;
      uint32_t* newU32 = reinterpret_cast<uint32_t*>(newBuffer);
      if (bufferFormat == BUFFER_FORMAT_OLD_VME) {
	//swap whole buffer
	for (size_t i = 0; i < sizeU32; i+=2) {
	  newU32[i] = originalU32[i+1];
	  newU32[i+1] = originalU32[i];
	}
      }
      if (bufferFormat == BUFFER_FORMAT_NEW) {
	//copy DAQ header
	memcpy(newU32,originalU32,8);
	//copy DAQ trailer
	memcpy(newU32+sizeU32-2,originalU32+sizeU32-2,8);
	//swap the payload
	for (size_t i = 2; i < sizeU32-2; i+=2) {
	  newU32[i] = originalU32[i+1];
	  newU32[i+1] = originalU32[i];
	}
      }
      orderedBuffer_ = newBuffer;
    } //if ( (bufferFormat == BUFFER_FORMAT_OLD_VME) || (bufferFormat == BUFFER_FORMAT_NEW) )
    else {
      orderedBuffer_ = originalBuffer_;
    }
  
    //construct header object at begining of buffer
    daqHeader_ = FEDDAQHeader(orderedBuffer_);
    //construct trailer object using last 64 bit word of buffer
    daqTrailer_ = FEDDAQTrailer(orderedBuffer_+bufferSize_-8);
  }

  FEDBufferBase::~FEDBufferBase()
  {
    //if the buffer was coppied and swapped then delete the copy
    if (orderedBuffer_ != originalBuffer_) delete[] orderedBuffer_;
  }

  void FEDBufferBase::print(std::ostream& os) const
  {
    os << "buffer format: " << bufferFormat() << std::endl;
    os << "Buffer size: " << bufferSize() << " bytes" << std::endl;
    os << "Event length from DAQ trailer: " << daqEventLengthInBytes() << " bytes" << std::endl;
    os << "Source ID: " << daqSourceID() << std::endl;
    os << "Header type: " << headerType() << std::endl;
    os << "Readout mode: " << readoutMode() << std::endl;
    os << "Data type: " << dataType() << std::endl;
    os << "DAQ event type: " << daqEventType() << std::endl;
    os << "TTS state: " << daqTTSState() << std::endl;
    os << "L1 ID: " << daqLvl1ID() << std::endl;
    os << "BX ID: " << daqBXID() << std::endl;
    os << "FED status register flags: "; fedStatusRegister().printFlags(os); os << std::endl;
    os << "APVe Address: " << uint16_t(apveAddress()) << std::endl;
    os << "Enabled FE units: " << uint16_t(nFEUnitsEnabled()) << std::endl;
  }

  uint8_t FEDBufferBase::nFEUnitsEnabled() const
  {
    uint8_t result = 0;
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (feEnabled(iFE)) result++;
    }
    return result;
  }

  bool FEDBufferBase::checkSourceIDs() const
  {
    return ( (daqSourceID() >= FED_ID_MIN) &&
	     (daqSourceID() <= FED_ID_MAX) );
  }
  
  bool FEDBufferBase::checkMajorityAddresses() const
  {
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!feEnabled(iFE)) continue;
      if (majorityAddressErrorForFEUnit(iFE)) return false;
    }
    return true;
  }
  
  bool FEDBufferBase::channelGood(const uint8_t internalFEDChannelNum) const
  {
    const uint8_t feUnit = internalFEDChannelNum/FEDCH_PER_FEUNIT;
    return ( !majorityAddressErrorForFEUnit(feUnit) && feEnabled(feUnit) && !feOverflow(feUnit) );
  }
  
  bool FEDBufferBase::doChecks() const
  {
    return (doTrackerSpecialHeaderChecks() && doDAQHeaderAndTrailerChecks());
  }

  std::string FEDBufferBase::checkSummary() const
  {
    std::stringstream summary;
    summary << "Check buffer type valid: " << ( checkBufferFormat() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check header format valid: " << ( checkHeaderType() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check readout mode valid: " << ( checkReadoutMode() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check APVe address valid: " << ( checkAPVEAddressValid() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check FE unit majority addresses: " << ( checkMajorityAddresses() ? "passed" : "FAILED" ) << std::endl;
    if (!checkMajorityAddresses()) {
      summary << "FEs with majority address error: ";
      unsigned int badFEs = 0;
      for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
	if (!feEnabled(iFE)) continue;
	if (majorityAddressErrorForFEUnit(iFE)) {
	  summary << uint16_t(iFE) << " ";
	  badFEs++;
	}
      }
      summary << std::endl;
      summary << "Number of FE Units with bad addresses: " << badFEs << std::endl;
    }
    summary << "Check for FE unit buffer overflows: " << ( checkNoFEOverflows() ? "passed" : "FAILED" ) << std::endl;
    if (!checkNoFEOverflows()) {
      summary << "FEs which overflowed: ";
      unsigned int badFEs = 0;
      for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
	if (feOverflow(iFE)) {
	  summary << uint16_t(iFE) << " ";
	  badFEs++;
	}
      }
      summary << std::endl;
      summary << "Number of FE Units which overflowed: " << badFEs << std::endl;
    }
    summary << "Check for S-Link CRC errors: " << ( checkNoSlinkCRCError() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check for S-Link transmission error: " << ( checkNoSLinkTransmissionError() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check CRC: " << ( checkCRC() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check source ID is FED ID: " << ( checkSourceIDs() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check for unexpected source ID at FRL: " << ( checkNoUnexpectedSourceID() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check there are no extra headers or trailers: " << ( checkNoExtraHeadersOrTrailers() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check length from trailer: " << ( checkLengthFromTrailer() ? "passed" : "FAILED" ) << std::endl;
    return summary.str();
  }




  void FEDRawChannelUnpacker::throwBadChannelLength(const uint16_t length)
  {
    std::stringstream ss;
    ss << "Channel length is invalid. Raw channels have 3 header bytes and 2 bytes per sample. "
       << "Channel length is " << uint16_t(length) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }




  void FEDZSChannelUnpacker::throwBadChannelLength(const uint16_t length)
  {
    std::stringstream ss;
    ss << "Channel length is longer than max allowed value. "
       << "Channel length is " << uint16_t(length) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDZSChannelUnpacker::throwBadClusterLength()
  {
    std::stringstream ss;
    ss << "Cluster does not fit into channel. "
       << "Cluster length is " << uint16_t(valuesLeftInCluster_) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

}
