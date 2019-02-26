#include <iomanip>
#include <ostream>
#include <sstream>
#include <cstring>

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"

namespace sistrip {

  FEDBuffer::FEDBuffer(const uint8_t* fedBuffer, const uint16_t fedBufferSize, const bool allowBadBuffer)
    : FEDBufferBase(fedBuffer,fedBufferSize,allowBadBuffer,false)
  {
    channels_.reserve(FEDCH_PER_FED);
    //build the correct type of FE header object
    if ( (headerType() != HEADER_TYPE_INVALID) && (headerType() != HEADER_TYPE_NONE) ) {
      feHeader_ = FEDFEHeader::newFEHeader(headerType(),getPointerToDataAfterTrackerSpecialHeader());
      payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader()+feHeader_->lengthInBytes();
    } else {
      feHeader_ = std::unique_ptr<FEDFEHeader>();
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
    if (readoutMode() == READOUT_MODE_SPY) {
      throw cms::Exception("FEDBuffer") << "Unpacking of spy channel data with FEDBuffer is not supported" << std::endl;
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
        channels_.insert(channels_.end(),uint16_t(FEDCH_PER_FED-validChannels_),FEDChannel(payloadPointer_,0,0));
      }
    }
  }

  FEDBuffer::~FEDBuffer()
  {
  }

  void FEDBuffer::findChannels()
  {
    //set min length to 2 for ZSLite, 7 for ZS and 3 for raw
    uint16_t minLength;
    switch (readoutMode()) {
    case READOUT_MODE_ZERO_SUPPRESSED:
    case READOUT_MODE_ZERO_SUPPRESSED_FAKE:
      minLength = 7;
      break;
    case READOUT_MODE_PREMIX_RAW:
      minLength = 2;
      break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE10:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE:
      minLength = 2;
      break;
    default:
      minLength = 3;
      break;
    }
    uint16_t offsetBeginningOfChannel = 0;
    for (uint16_t i = 0; i < FEDCH_PER_FED; i++) {
      //if FE unit is not enabled then skip rest of FE unit adding NULL pointers
      if UNLIKELY( !(fePresent(i/FEDCH_PER_FEUNIT) && feEnabled(i/FEDCH_PER_FEUNIT)) ) {
	channels_.insert(channels_.end(),uint16_t(FEDCH_PER_FEUNIT),FEDChannel(payloadPointer_,0,0));
	i += FEDCH_PER_FEUNIT-1;
	validChannels_ += FEDCH_PER_FEUNIT;
	continue;
      }
      //if FE unit is enabled
      //check that channel length bytes fit into buffer
      if UNLIKELY(offsetBeginningOfChannel+1 >= payloadLength_) {
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

      //check that the channel length is long enough to contain the header
      if UNLIKELY(channelLength < minLength) {
        SiStripFedKey key(0,i/FEDCH_PER_FEUNIT,i%FEDCH_PER_FEUNIT);
        std::ostringstream ss;
        ss << "Channel " << uint16_t(i) << " (FE unit " << key.feUnit() << " channel " << key.feChan() << " according to external numbering scheme)"
           << " is too short. "
           << "Channel starts at " << uint16_t(offsetBeginningOfChannel) << " in payload. "
           << "Channel length is " << uint16_t(channelLength) << ". "
           << "Min length is " << uint16_t(minLength) << ". ";
        throw cms::Exception("FEDBuffer") << ss.str();
      }
      if UNLIKELY(offsetBeginningOfChannel+channelLength > payloadLength_) {
        SiStripFedKey key(0,i/FEDCH_PER_FEUNIT,i%FEDCH_PER_FEUNIT);
	std::ostringstream ss;
        ss << "Channel " << uint16_t(i) << " (FE unit " << key.feUnit() << " channel " << key.feChan() << " according to external numbering scheme)" 
           << "Channel starts at " << uint16_t(offsetBeginningOfChannel) << " in payload. "
           << "Channel length is " << uint16_t(channelLength) << ". "
           << "Payload length is " << uint16_t(payloadLength_) << ". ";
        throw cms::Exception("FEDBuffer") << ss.str();
      }

      validChannels_++;
      const uint16_t offsetEndOfChannel = offsetBeginningOfChannel+channelLength;
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
  
  bool FEDBuffer::channelGood(const uint8_t internalFEDChannelNum, const bool doAPVeCheck) const
  {
    return ( (internalFEDChannelNum < validChannels_) &&
	     ( (doAPVeCheck && feGood(internalFEDChannelNum/FEDCH_PER_FEUNIT)) || 
	       (!doAPVeCheck && feGoodWithoutAPVEmulatorCheck(internalFEDChannelNum/FEDCH_PER_FEUNIT)) 
	       ) &&
             (this->readoutMode() == sistrip::READOUT_MODE_SCOPE || checkStatusBits(internalFEDChannelNum)) );
  }

  bool FEDBuffer::doChecks(bool doCRC) const
  {
    //check that all channels were unpacked properly
    if (validChannels_ != FEDCH_PER_FED) return false;
    //do checks from base class
    if (!FEDBufferBase::doChecks()) return false;
    //check CRC
    if (doCRC  &&  !checkCRC()) return false;
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
    const uint16_t payloadLengthInWords = payloadLength_/8;
  
    //find channel length
    //find last enabled FE unit
    uint8_t lastEnabledFeUnit = 7;
    while ( !(fePresent(lastEnabledFeUnit) && feEnabled(lastEnabledFeUnit)) && lastEnabledFeUnit!=0 ) lastEnabledFeUnit--;
    //last channel is last channel on last enabled FE unit
    const FEDChannel& lastChannel = channels_[internalFEDChannelNum(lastEnabledFeUnit,FEDCH_PER_FEUNIT-1)];
    const uint16_t offsetLastChannel = lastChannel.offset();
    const uint16_t offsetEndOfChannelData = offsetLastChannel+lastChannel.length();
    const uint16_t channelDataLength = offsetEndOfChannelData;
    //channel length in words is length in bytes rounded up to nearest word
    uint16_t channelDataLengthInWords = channelDataLength/8;
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
      if (FEDBuffer::channelGood(iCh, true)) {
	//if a channel is bad then return false
	if (channels_[iCh].packetCode() != correctPacketCode) return false;
      }
    }
    //if no bad channels were found the they are all ok
    return true;
  }

  bool FEDBuffer::checkFEUnitAPVAddresses() const
  {
    //get golden address
    const uint8_t goldenAddress = apveAddress();
    //don't check if the address is 00 since APVe is probably not connected
    if (goldenAddress == 0x00) return true;
    //check can only be done for full debug headers
    const FEDFullDebugHeader* fdHeader = dynamic_cast<FEDFullDebugHeader*>(feHeader_.get());
    if (!fdHeader) return true;
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
    std::ostringstream summary;
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




  void FEDRawChannelUnpacker::throwBadChannelLength(const uint16_t length)
  {
    std::ostringstream ss;
    ss << "Channel length is invalid. Raw channels have 3 header bytes and 2 bytes per sample. "
       << "Channel length is " << uint16_t(length) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDBSChannelUnpacker::throwBadChannelLength(const uint16_t length)
  {
    std::ostringstream ss;
    ss << "Channel length is invalid. "
       << "Channel length is " << uint16_t(length) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDBSChannelUnpacker::throwBadWordLength(const uint16_t word_length)
  {
    std::ostringstream ss;
    ss << "Word length is invalid. "
       << "Word length is " << word_length << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDBSChannelUnpacker::throwUnorderedData(const uint8_t currentStrip, const uint8_t firstStripOfNewCluster)
  {
    std::ostringstream ss;
    ss << "First strip of new cluster is not greater than last strip of previous cluster. "
       << "Last strip of previous cluster is " << uint16_t(currentStrip) << ". "
       << "First strip of new cluster is " << uint16_t(firstStripOfNewCluster) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDZSChannelUnpacker::throwBadChannelLength(const uint16_t length)
  {
    std::ostringstream ss;
    ss << "Channel length is longer than max allowed value. "
       << "Channel length is " << uint16_t(length) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDZSChannelUnpacker::throwBadClusterLength()
  {
    std::ostringstream ss;
    ss << "Cluster does not fit into channel. "
       << "Cluster length is " << uint16_t(valuesLeftInCluster_) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }
  
  void FEDZSChannelUnpacker::throwUnorderedData(const uint8_t currentStrip, const uint8_t firstStripOfNewCluster)
  {
    std::ostringstream ss;
    ss << "First strip of new cluster is not greater than last strip of previous cluster. "
       << "Last strip of previous cluster is " << uint16_t(currentStrip) << ". "
       << "First strip of new cluster is " << uint16_t(firstStripOfNewCluster) << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

}
