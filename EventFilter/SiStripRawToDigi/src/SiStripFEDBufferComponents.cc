#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include <iomanip>
#include <ostream>

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


  std::ostream& operator<<(std::ostream& os, const FEDBufferFormat& value)
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

  FEDBufferFormat TrackerSpecialHeader::bufferFormat() const
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
	     !apvErrorFromBit(internalFEDChannelNum,0) &&
	     !apvAddressErrorFromBit(internalFEDChannelNum,0) &&
	     !apvErrorFromBit(internalFEDChannelNum,1) &&
	     !apvAddressErrorFromBit(internalFEDChannelNum,1) );
  }

  FEDFEHeader::~FEDFEHeader()
  {
  }

}
