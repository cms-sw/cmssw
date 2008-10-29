#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include <iomanip>
#include <sstream>

namespace sistrip {

  void printHexValue(uint8_t value, std::ostream& os)
  {
    std::ios_base::fmtflags originalFormatFlags = os.flags();
    os << std::hex << std::setfill('0') << std::setw(2);
    os << uint16_t(value);
    os.flags(originalFormatFlags);
  }

  void printHexWord(const uint8_t* pointer, size_t lengthInBytes, std::ostream& os)
  {
    size_t i = lengthInBytes-1;
    do{
      printHexValue(pointer[i],os);
      if (i != 0) os << " ";
    } while (i-- != 0);
  }

  void printHex(const void* pointer, size_t lengthInBytes, std::ostream& os)
  {
    const uint8_t* bytePointer = reinterpret_cast<const uint8_t*>(pointer);
    //if there is one 64 bit word or less, print it out
    if (lengthInBytes <= 8) {
      printHexWord(bytePointer,lengthInBytes,os);
    }
    //otherwise, print word numbers etc
    else {
      //header
      os << "word\tbyte\t                       \t\tbyte" << std::endl;;
      size_t words = lengthInBytes/8;
      size_t extraBytes = lengthInBytes - 8*words;
      //print full words
      for (size_t w = 0; w < words; w++) {
	const size_t startByte = w*8;
	os << w << '\t' << startByte+8 << '\t';
	printHexWord(bytePointer+startByte,8,os);
	os << "\t\t" << startByte << std::endl;
      }
      //print part word, if any
      if (extraBytes) {
	const size_t startByte = words*8;
	os << words << '\t' << startByte+8 << '\t';
	//padding
	size_t p = 8;
	while (p-- > extraBytes) {
	  os << "00 ";
	}
	printHexWord(bytePointer+startByte,extraBytes,os);
	os << "\t\t" << startByte << std::endl;
      }
      os << std::endl;
    }
  }


  std::ostream& operator<<(std::ostream& os, const BufferFormat& value)
  {
    switch (value) {
    case BUFFER_FORMAT_OLD_VME:
      os << "Old VME";
      break;
    case BUFFER_FORMAT_OLD_SLINK:
      os << "Old S-Link";
      break;
    case BUFFER_FORMAT_NEW:
      os << "New";
      break;
    case BUFFER_FORMAT_INVALID:
      os << "Invalid";
      break;
    default:
      os << "Unrecognized";
      break;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const FEDHeaderType& value)
  {
    switch (value) {
    case HEADER_TYPE_FULL_DEBUG:
      os << "Full debug";
      break;
    case HEADER_TYPE_APV_ERROR:
      os << "APV error";
      break;
    case HEADER_TYPE_INVALID:
      os << "Invalid";
      break;
    default:
      os << "Unrecognized";
      break;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const FEDReadoutMode& value)
  {
    switch (value) {
    case READOUT_MODE_SCOPE:
      os << "Scope mode";
      break;
    case READOUT_MODE_VIRGIN_RAW:
      os << "Virgin raw";
      break;
    case READOUT_MODE_PROC_RAW:
      os << "Processed raw";
      break;
    case READOUT_MODE_ZERO_SUPPRESSED:
      os << "Zero suppressed";
      break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE:
      os << "Zero suppressed lite";
      break;
    case READOUT_MODE_INVALID:
      os << "Invalid";
      break;
    default:
      os << "Unrecognized";
      break;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const FEDDataType& value)
  {
    switch (value) {
    case DATA_TYPE_REAL:
      os << "Real data";
      break;
    case DATA_TYPE_FAKE:
      os << "Fake data";
      break;
    default:
      os << "Unrecognized";
      break;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const FEDDAQEventType& value)
  {
    switch (value) {
    case DAQ_EVENT_TYPE_PHYSICS:
      os << "Physics trigger";
      break;
    case DAQ_EVENT_TYPE_CALIBRATION:
      os << "Calibration trigger";
      break;
    case DAQ_EVENT_TYPE_TEST:
      os << "Test trigger";
      break;
    case DAQ_EVENT_TYPE_TECHNICAL:
      os << "Technical trigger";
      break;
    case DAQ_EVENT_TYPE_SIMULATED:
      os << "Simulated event";
      break;
    case DAQ_EVENT_TYPE_TRACED:
      os << "Traced event";
      break;
    case DAQ_EVENT_TYPE_ERROR:
      os << "Error";
      break;
    case DAQ_EVENT_TYPE_INVALID:
      os << "Unknown";
      break;
    default:
      os << "Unrecognized";
      break;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const FEDTTSBits& value)
  {
    switch (value) {
    case TTS_DISCONNECTED1:
      os << "Disconected 1";
      break;
    case TTS_WARN_OVERFLOW:
      os << "Warning overflow";
      break;
    case TTS_OUT_OF_SYNC:
      os << "Out of sync";
      break;
    case TTS_BUSY:
      os << "Busy";
      break;
    case TTS_READY:
      os << "Ready";
      break;
    case TTS_ERROR:
      os << "Error";
      break;
    case TTS_INVALID:
      os << "Invalid";
      break;
    case TTS_DISCONNECTED2:
      os << "Disconected 2";
      break;
    default:
      os << "Unrecognized";
      break;
    }
    return os;
  }




  FEDBuffer::FEDBuffer(const uint8_t* fedBuffer, size_t fedBufferSize, bool allowBadBuffer)
    : FEDBufferBase(fedBuffer,fedBufferSize,allowBadBuffer)
  {
    channels_.reserve(CHANNELS_PER_FED);
    payloadPointer_ = NULL;
    //build the correct type of FE header object
    if (headerType() != HEADER_TYPE_INVALID) {
      feHeader_ = FEDFEHeader::newFEHeader(headerType(),getPointerToDataAfterTrackerSpecialHeader());
      //set the address of the start of the data for the first channel using the length from the feHeader object
      payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader()+feHeader_->lengthInBytes();
    } else {
      feHeader_ = NULL;
      payloadPointer_ = getPointerToDataAfterTrackerSpecialHeader();
      if (!allowBadBuffer) {
	std::ostringstream ss;
	ss << "Header type is invalid. "
	   << "Header type nibble is ";
	uint8_t headerNibble = trackerSpecialHeader().headerTypeNibble();
	printHex(&headerNibble,1,ss);
	ss << ". ";
	throw cms::Exception("FEDBuffer") << ss;
      }
    }
    lastValidChannel_ = 0;
    payloadLength_ = getPointerToByteAfterEndOfPayload()-payloadPointer_;
    //try to find channels
    try {
      findChannels();
    } catch (const cms::Exception& e) {
      //if there was a problem either rethrow the exception or just mark channel pointers NULL
      if (!allowBadBuffer) throw;
      else {
	channels_.insert(channels_.end(),size_t(CHANNELS_PER_FED-lastValidChannel_),FEDChannel(payloadPointer_,0));
      }
    }
  }

  FEDBuffer::~FEDBuffer()
  {
    if (feHeader_) delete feHeader_;
  }

  void FEDBuffer::findChannels()
  {
    size_t offsetBeginningOfChannel = 0;
    for (size_t i = 0; i < CHANNELS_PER_FED; i++) {
      //if FE unit is not enabled then skip rest of FE unit adding NULL pointers
      if (!feEnabled(i/CHANNELS_PER_FEUNIT)) {
	channels_.insert(channels_.end(),size_t(CHANNELS_PER_FEUNIT),FEDChannel(payloadPointer_,0));
	i += CHANNELS_PER_FEUNIT-1;
	lastValidChannel_ += CHANNELS_PER_FEUNIT;
	continue;
      }
      //if FE unit is enabled
      //check that channel length bytes fit into buffer
      if (offsetBeginningOfChannel+2 >= payloadLength_) {
	throw cms::Exception("FEDBuffer") << "Channel " << uint16_t(i) << " does not fit into buffer. "
						 << "Channel starts at " << uint16_t(offsetBeginningOfChannel) << " in payload. "
						 << "Payload length is " << uint16_t(payloadLength_) << ". ";
      }
      channels_.push_back(FEDChannel(payloadPointer_,offsetBeginningOfChannel));
      //get length and check that whole channel fits into buffer
      uint16_t channelLength = channels_.back().length();
      if (offsetBeginningOfChannel+channelLength > payloadLength_) {
	throw cms::Exception("FEDBuffer") << "Channel " << uint16_t(i) << " does not fit into buffer. "
						 << "Channel starts at " << uint16_t(offsetBeginningOfChannel) << " in payload. "
						 << "Channel length is " << uint16_t(channelLength) << ". "
						 << "Payload length is " << uint16_t(payloadLength_) << ". ";
      }
      lastValidChannel_++;
      const size_t offsetEndOfChannel = offsetBeginningOfChannel+channelLength;
      //add padding if necessary and calculate offset for begining of next channel
      if (!( (i+1) % CHANNELS_PER_FEUNIT )) {
	uint8_t numPaddingBytes = 8 - (offsetEndOfChannel % 8);
	if (numPaddingBytes == 8) numPaddingBytes = 0;
	offsetBeginningOfChannel = offsetEndOfChannel + numPaddingBytes;
      } else {
	offsetBeginningOfChannel = offsetEndOfChannel;
      }
    }
  }

  bool FEDBuffer::doChecks() const
  {
    //check that all channels were unpacked properly
    if (lastValidChannel_ != CHANNELS_PER_FED) return false;
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
    for (uint8_t iCh = 0; iCh < CHANNELS_PER_FED; iCh++) {
      //if FE unit is disabled then skip all channels on it
      if (!feEnabled(iCh/CHANNELS_PER_FED)) {
	iCh += CHANNELS_PER_FEUNIT-1;
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
    return (lastValidChannel_ == CHANNELS_PER_FED);
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
    while (!feEnabled(lastEnabledFeUnit)) lastEnabledFeUnit--;
    //last channel is last channel on last enabled FE unit
    const FEDChannel& lastChannel = channels_[internalFEDChannelNum(lastEnabledFeUnit,CHANNELS_PER_FEUNIT-1)];
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
    //if mode is ZS Lite then retyurn true since check can't be done since packet code is missing
    //for other modes get the correct code
    //if the mode is not valid then return false
    uint8_t correctPacketCode = 0x00;
    switch (readoutMode()) {
    case READOUT_MODE_ZERO_SUPPRESSED_LITE:
      return true;
    case READOUT_MODE_SCOPE:
      correctPacketCode = PACKET_CODE_SCOPE;
      break;
    case READOUT_MODE_VIRGIN_RAW:
      correctPacketCode = PACKET_CODE_VIRGIN_RAW;
      break;
    case READOUT_MODE_PROC_RAW:
      correctPacketCode = PACKET_CODE_PROC_RAW;
      break;
    case READOUT_MODE_ZERO_SUPPRESSED:
      correctPacketCode = PACKET_CODE_ZERO_SUPPRESSED;
      break;
    default:
      return false;
    }
    for (uint8_t iCh = 0; iCh < CHANNELS_PER_FED; iCh++) {
      //if FE unit is disabled then skip all channels on it
      if (!feEnabled(iCh/CHANNELS_PER_FED)) {
	iCh += CHANNELS_PER_FEUNIT-1;
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
    const FEDFullDebugHeader* fdHeader = dynamic_cast<const FEDFullDebugHeader*>(feHeader_);
    if (!fdHeader) return true;
    //get golden address
    const uint8_t goldenAddress = apveAddress();
    //check all enabled FE units
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!feEnabled(iFE)) continue;
      //if address is bad then return false
      if (fdHeader->feUnitMajorityAddress(iFE) != goldenAddress) return false;
    }
    //if no bad addresses were found then return true
    return true;
  }

  bool FEDBuffer::checkFEUnitLengths() const
  {
    //check can only be done for full debug headers
    const FEDFullDebugHeader* fdHeader = dynamic_cast<const FEDFullDebugHeader*>(feHeader_);
    if (!fdHeader) return true;
    //check lengths for enabled FE units
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!feEnabled(iFE)) continue;
      //get length from channels
      uint16_t lengthFromChannels = 0;
      for (uint8_t iCh = 0; iCh < CHANNELS_PER_FEUNIT; iCh++) {
	lengthFromChannels += channels_[internalFEDChannelNum(iFE,iCh)].length();
      }
      //round to nearest 64bit word
      if (lengthFromChannels % 8) lengthFromChannels = (lengthFromChannels & ~0x7) + 7;
      if (lengthFromChannels != fdHeader->feUnitLength(iFE)) return false;
    }
    //if no errors were encountered then return true
    return true;
  }

  std::string FEDBuffer::checkSummary() const
  {
    std::stringstream summary;
    summary << FEDBufferBase::checkSummary();
    summary << "Check channel status bits: " << ( checkAllChannelStatusBits() ? "passed" : "FAILED" ) << std::endl;
    if (!checkAllChannelStatusBits()) {
      summary << "Channels with errors: ";
      unsigned int badChannels = 0;
      for (uint8_t iCh = 0; iCh < CHANNELS_PER_FED; iCh++) {
	if (!feEnabled(iCh/CHANNELS_PER_FED)) continue;
	if (!checkStatusBits(iCh)) {
	  summary << uint16_t(iCh) << " ";
	  badChannels++;
	}
      }
      summary << std::endl;
      summary << "Number of channels with bad status bits: " << badChannels << std::endl;
    }
    summary << "Check channel lengths match buffer length: " << ( checkChannelLengthsMatchBufferLength() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check channel packet codes: " << ( checkChannelPacketCodes() ? "passed" : "FAILED" ) << std::endl;
    //summary << "Check cluster lengths: " << ( checkClusterLengths() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check FE unit lengths: " << ( checkFEUnitLengths() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check FE unit APV addresses match APVe: " << ( checkFEUnitAPVAddresses() ? "passed" : "FAILED" ) << std::endl;
    return summary.str();
  }




  FEDBufferBase::FEDBufferBase(const uint8_t* fedBuffer, size_t fedBufferSize, bool allowUnrecognizedFormat)
    : originalBuffer_(fedBuffer),
      bufferSize_(fedBufferSize)
  {
    //min buffer length. DAQ header, DAQ trailer, tracker special header. 
    static const size_t MIN_BUFFER_SIZE = 8+8+8;
    //check size is non zero and data pointer is not NULL
    if (!originalBuffer_) throw cms::Exception("FEDBuffer") << "Buffer pointer is NULL. ";
    if (bufferSize_ < MIN_BUFFER_SIZE) 
      throw cms::Exception("FEDBuffer") << "Buffer is too small. "
					       << "Min size is " << MIN_BUFFER_SIZE << ". "
					       << "Buffer size is " << bufferSize_ << ". ";
  
    //construct tracker special header using second 64 bit word
    specialHeader_ = TrackerSpecialHeader(originalBuffer_+8);
  
    //check the buffer format
    const BufferFormat bufferFormat = specialHeader_.bufferFormat();
    if (bufferFormat == BUFFER_FORMAT_INVALID && !allowUnrecognizedFormat) {
      cms::Exception e("FEDBuffer");
      e << "Buffer format not recognized. "
	<< "Tracker special header: " << specialHeader_;
      throw e;
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

  uint16_t FEDBufferBase::calcCRC() const
  {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < bufferSize_-8; i++) {
      crc = evf::compute_crc_8bit(crc,orderedBuffer_[i^7]);
    }
    for (size_t i=bufferSize_-8; i<bufferSize_; i++) {
      uint8_t byte;
      //set CRC bytes to zero since these were not set when CRC was calculated
      if (i==bufferSize_-4 || i==bufferSize_-3)
	byte = 0x00;
      else
	byte = orderedBuffer_[i^7];
      crc = evf::compute_crc_8bit(crc,byte);
    }
    return crc;
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
    os << checkSummary();
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
    return ( (daqSourceID() >= FEDNumbering::getSiStripFEDIds().first) &&
	     (daqSourceID() <= FEDNumbering::getSiStripFEDIds().second) );
  }

  bool FEDBufferBase::checkMajorityAddresses() const
  {
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      if (!feEnabled(iFE)) continue;
      if (majorityAddressErrorForFEUnit(iFE)) return false;
    }
    return true;
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
    summary << "Check for S-Link CRC errors: " << ( checkNoSlinkCRCError() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check for S-Link transmission error: " << ( checkNoSLinkTransmissionError() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check CRC: " << ( checkCRC() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check source ID is FED ID: " << ( checkSourceIDs() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check for unexpected source ID at FRL: " << ( checkNoUnexpectedSourceID() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check there are no extra headers or trailers: " << ( checkNoExtraHeadersOrTrailers() ? "passed" : "FAILED" ) << std::endl;
    summary << "Check length from trailer: " << ( checkLengthFromTrailer() ? "passed" : "FAILED" ) << std::endl;
    return summary.str();
  }




  void FEDStatusRegister::printFlags(std::ostream& os) const
  {
    if (slinkFullFlag()) os << "SLINK_FULL ";
    if (trackerHeaderMonitorDataReadyFlag()) os << "HEADER_MONITOR_READY ";
    if (qdrMemoryFullFlag()) os << "QDR_FULL ";
    if (qdrMemoryPartialFullFlag()) os << "QDR_PARTIAL_FULL ";
    if (qdrMemoryEmptyFlag()) os << "QDR_EMPTY ";
    if (l1aBxFIFOFullFlag()) os << "L1A_FULL ";
    if (l1aBxFIFOPartialFullFlag()) os << "L1A_PARTIAL_FULL ";
    if (l1aBxFIFOEmptyFlag()) os << "L1A_EMPTY ";
  }




  void FEDBackendStatusRegister::printFlags(std::ostream& os) const
  {
    if (internalFreezeFlag()) os << "INTERNAL_FREEZE ";
    if (slinkDownFlag()) os << "SLINK_DOWN ";
    if (slinkFullFlag()) os << "SLINK_FULL ";
    if (backpressureFlag()) os << "BACKPRESSURE ";
    if (ttcReadyFlag()) os << "TTC_READY ";
    if (trackerHeaderMonitorDataReadyFlag()) os << "HEADER_MONITOR_READY ";
    if (qdrMemoryFullFlag()) os << "QDR_FULL ";
    if (qdrMemoryPartialFullFlag()) os << "QDR_PARTIAL_FULL ";
    if (qdrMemoryEmptyFlag()) os << "QDR_EMPTY ";
    if (frameAddressFIFOFullFlag()) os << "FRAME_ADDRESS_FULL ";
    if (frameAddressFIFOPartialFullFlag()) os << "FRAME_ADDRESS_PARTIAL_FULL ";
    if (frameAddressFIFOEmptyFlag()) os << "FRAME_ADDRESS_EMPTY ";
    if (totalLengthFIFOFullFlag()) os << "TOTAL_LENGTH_FULL ";
    if (totalLengthFIFOPartialFullFlag()) os << "TOTAL_LENGTH_PARTIAL_FULL ";
    if (totalLengthFIFOEmptyFlag()) os << "TOTAL_LENGTH_EMPTY ";
    if (trackerHeaderFIFOFullFlag()) os << "TRACKER_HEADER_FULL ";
    if (trackerHeaderFIFOPartialFullFlag()) os << "TRACKER_HEADER_PARTIAL_FULL ";
    if (trackerHeaderFIFOEmptyFlag()) os << "TRACKER_HEADER_EMPTY ";
    if (l1aBxFIFOFullFlag()) os << "L1A_FULL ";
    if (l1aBxFIFOPartialFullFlag()) os << "L1A_PARTIAL_FULL ";
    if (l1aBxFIFOEmptyFlag()) os << "L1A_EMPTY ";
    if (feEventLengthFIFOFullFlag()) os << "FE_LENGTH_FULL ";
    if (feEventLengthFIFOPartialFullFlag()) os << "FE_LENGTH_PARTIAL_FULL ";
    if (feEventLengthFIFOEmptyFlag()) os << "FE_LENGTH_EMPTY ";
    if (feFPGAFullFlag()) os << "FE_FPGA_FULL ";
    if (feFPGAPartialFullFlag()) os << "FE_FPGA_PARTIAL_FULL ";
    if (feFPGAEmptyFlag()) os << "FE_FPGA_EMPTY ";
  }




  TrackerSpecialHeader::TrackerSpecialHeader(const uint8_t* headerPointer)
  {
    //the buffer format byte is one of the valid values if we assume the buffer is not swapped
    const bool validFormatByteWhenNotWordSwapped = ( (headerPointer[BUFFERFORMAT] == BUFFER_FORMAT_CODE_NEW) ||
						     (headerPointer[BUFFERFORMAT] == BUFFER_FORMAT_CODE_OLD) );
    //the buffer format byte is the old value if we assume the buffer is swapped
    const bool validFormatByteWhenWordSwapped = (headerPointer[BUFFERFORMAT^4] == BUFFER_FORMAT_CODE_OLD);
    //if the buffer format byte is valid if the buffer is not swapped or it is never valid
    if (validFormatByteWhenNotWordSwapped || (!validFormatByteWhenNotWordSwapped && !validFormatByteWhenWordSwapped) ) {
      memcpy(specialHeader_,headerPointer,8);
      wordSwapped_ = false;
    } else {
      memcpy(specialHeader_,headerPointer+4,4);
      memcpy(specialHeader_+4,headerPointer,4);
      wordSwapped_ = true;
    }
  }

  BufferFormat TrackerSpecialHeader::bufferFormat() const
  {
    if (bufferFormatByte() == BUFFER_FORMAT_CODE_NEW) return BUFFER_FORMAT_NEW;
    else if (bufferFormatByte() == BUFFER_FORMAT_CODE_OLD) {
      if (wordSwapped_) return BUFFER_FORMAT_OLD_VME;
      else return BUFFER_FORMAT_OLD_SLINK;
    }
    else return BUFFER_FORMAT_INVALID;
  }

  FEDHeaderType TrackerSpecialHeader::headerType() const
  {
    if ( (headerTypeNibble() == HEADER_TYPE_FULL_DEBUG) || 
	 (headerTypeNibble() == HEADER_TYPE_APV_ERROR) )
      return FEDHeaderType(headerTypeNibble());
    else return HEADER_TYPE_INVALID;
  }

  FEDReadoutMode TrackerSpecialHeader::readoutMode() const
  {
    const uint8_t eventTypeNibble = trackerEventTypeNibble();
    //if it is scope mode then return as is (it cannot be fake data)
    if (eventTypeNibble == READOUT_MODE_SCOPE) return FEDReadoutMode(eventTypeNibble);
    //if not then ignore the last bit which indicates if it is real or fake
    else {
      const uint8_t mode = (eventTypeNibble & 0xE);
      switch(mode) {
      case READOUT_MODE_VIRGIN_RAW:
      case READOUT_MODE_PROC_RAW:
      case READOUT_MODE_ZERO_SUPPRESSED:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE:
	return FEDReadoutMode(mode);
      default:
	return READOUT_MODE_INVALID;
      }
    }   
  }

  FEDDataType TrackerSpecialHeader::dataType() const
  {
    uint8_t eventTypeNibble = trackerEventTypeNibble();
    //if it is scope mode then it is always real
    if (eventTypeNibble == READOUT_MODE_SCOPE) return DATA_TYPE_REAL;
    //in other modes it is the lowest order bit of event type nibble
    else return FEDDataType(eventTypeNibble & 0x1);
  }




  FEDDAQEventType FEDDAQHeader::eventType() const
  {
    switch(eventTypeNibble()) {
    case DAQ_EVENT_TYPE_PHYSICS:
    case DAQ_EVENT_TYPE_CALIBRATION:
    case DAQ_EVENT_TYPE_TEST:
    case DAQ_EVENT_TYPE_TECHNICAL:
    case DAQ_EVENT_TYPE_SIMULATED:
    case DAQ_EVENT_TYPE_TRACED:
    case DAQ_EVENT_TYPE_ERROR:
      return FEDDAQEventType(eventTypeNibble());
    default:
      return DAQ_EVENT_TYPE_INVALID;
    }
  }




  FEDTTSBits FEDDAQTrailer::ttsBits() const
  {
    switch(ttsNibble()) {
    case TTS_DISCONNECTED1:
    case TTS_WARN_OVERFLOW:
    case TTS_OUT_OF_SYNC:
    case TTS_BUSY:
    case TTS_READY:
    case TTS_ERROR:
    case TTS_DISCONNECTED2:
      return FEDTTSBits(ttsNibble());
    default:
      return TTS_INVALID;
    }
  }




  FEDAPVErrorHeader::~FEDAPVErrorHeader()
  {
  }

  size_t FEDAPVErrorHeader::lengthInBytes() const
  {
    return APV_ERROR_HEADER_SIZE_IN_BYTES;
  }

  void FEDAPVErrorHeader::print(std::ostream& os) const
  {
    printHex(header_,APV_ERROR_HEADER_SIZE_IN_BYTES,os);
  }

  bool FEDAPVErrorHeader::checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const
  {
    //TODO: check ordering
    uint8_t byteNumber = internalFEDChannelNum * 2 / 8;
    uint8_t bitInByte = internalFEDChannelNum * 2 % 8;
    //bit high means no error
    return (!(header_[byteNumber] & (0x1<<bitInByte) ));
  }

  bool FEDAPVErrorHeader::checkChannelStatusBits(uint8_t internalFEDChannelNum) const
  {
    return (checkStatusBits(internalFEDChannelNum,0) && checkStatusBits(internalFEDChannelNum,1));
  }


  FEDFullDebugHeader::~FEDFullDebugHeader()
  {
  }

  size_t FEDFullDebugHeader::lengthInBytes() const
  {
    return FULL_DEBUG_HEADER_SIZE_IN_BYTES;
  }

  void FEDFullDebugHeader::print(std::ostream& os) const
  {
    printHex(header_,FULL_DEBUG_HEADER_SIZE_IN_BYTES,os);
  }

  bool FEDFullDebugHeader::checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const
  {
    return ( !unlockedFromBit(internalFEDChannelNum) &&
	     !outOfSyncFromBit(internalFEDChannelNum) &&
	     !apvError(internalFEDChannelNum,apvNum) &&
	     !apvAddressError(internalFEDChannelNum,apvNum) );
  }

  bool FEDFullDebugHeader::checkChannelStatusBits(uint8_t internalFEDChannelNum) const
  {
    return ( !unlockedFromBit(internalFEDChannelNum) &&
	     !outOfSyncFromBit(internalFEDChannelNum) &&
	     !apvError(internalFEDChannelNum,0) &&
	     !apvAddressError(internalFEDChannelNum,0) &&
	     !apvError(internalFEDChannelNum,1) &&
	     !apvAddressError(internalFEDChannelNum,1) );
  }

  FEDFEHeader::~FEDFEHeader()
  {
  }




  void FEDRawChannelUnpacker::throwBadChannelLength(uint16_t length)
  {
    std::stringstream ss;
    ss << "Channel length is invalid. Raw channels have 3 header bytes and 2 bytes per sample. "
       << "Channel length is " << length << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }




  void FEDZSChannelUnpacker::throwBadChannelLength(uint16_t length)
  {
    std::stringstream ss;
    ss << "Channel length is longer than max allowed value. "
       << "Channel length is " << length << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

  void FEDZSChannelUnpacker::throwBadClusterLength()
  {
    std::stringstream ss;
    ss << "Cluster does not fit into channel. "
       << "Cluster length is " << valuesLeftInCluster_ << "."
       << std::endl;
    throw cms::Exception("FEDBuffer") << ss.str();
  }

}
