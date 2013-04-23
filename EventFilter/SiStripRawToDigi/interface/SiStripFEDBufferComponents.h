#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBufferComponents_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBufferComponents_H

#include "boost/cstdint.hpp"
#include <ostream>
#include <memory>
#include <cstring>
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace sistrip {
  
  //
  // Constants
  //
  
  static const uint8_t INVALID=0xFF;

  static const uint8_t APV_MAX_ADDRESS=192;
  
  static const uint16_t SCOPE_MODE_MAX_SCOPE_LENGTH=1022;

  enum FEDBufferFormat { BUFFER_FORMAT_INVALID=INVALID,
                         BUFFER_FORMAT_OLD_VME,
                         BUFFER_FORMAT_OLD_SLINK,
                         BUFFER_FORMAT_NEW
                       };
  //these are the values which appear in the buffer.
  static const uint8_t BUFFER_FORMAT_CODE_OLD = 0xED;
  static const uint8_t BUFFER_FORMAT_CODE_NEW = 0xC5;

  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDHeaderType { HEADER_TYPE_INVALID=INVALID,
                       HEADER_TYPE_FULL_DEBUG=1,
                       HEADER_TYPE_APV_ERROR=2,
                       HEADER_TYPE_NONE=4 //spy channel
                     };

  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDReadoutMode { READOUT_MODE_INVALID=INVALID,
                        READOUT_MODE_SCOPE=0x1,
                        READOUT_MODE_VIRGIN_RAW=0x2,
                        READOUT_MODE_PROC_RAW=0x6,
                        READOUT_MODE_ZERO_SUPPRESSED=0xA,
                        READOUT_MODE_ZERO_SUPPRESSED_LITE=0xC,
                        READOUT_MODE_SPY=0xE
                      };

  static const uint8_t PACKET_CODE_SCOPE = 0xE5;
  static const uint8_t PACKET_CODE_VIRGIN_RAW = 0xE6;
  static const uint8_t PACKET_CODE_PROC_RAW = 0xF2;
  static const uint8_t PACKET_CODE_ZERO_SUPPRESSED = 0xEA;

  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDDataType { DATA_TYPE_REAL=0,
                     DATA_TYPE_FAKE=1
                   };

  //enum values are values which appear in buffer. DO NOT CHANGE!
  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  enum FEDDAQEventType { DAQ_EVENT_TYPE_PHYSICS=0x1,
                         DAQ_EVENT_TYPE_CALIBRATION=0x2,
                         DAQ_EVENT_TYPE_TEST=0x3,
                         DAQ_EVENT_TYPE_TECHNICAL=0x4,
                         DAQ_EVENT_TYPE_SIMULATED=0x5,
                         DAQ_EVENT_TYPE_TRACED=0x6,
                         DAQ_EVENT_TYPE_ERROR=0xF,
                         DAQ_EVENT_TYPE_INVALID=INVALID
                       };

  //enum values are values which appear in buffer. DO NOT CHANGE!
  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  enum FEDTTSBits { TTS_DISCONNECTED0=0x0,
                    TTS_WARN_OVERFLOW=0x1,
                    TTS_OUT_OF_SYNC=0x2,
                    TTS_BUSY=0x4,
                    TTS_READY=0x8,
                    TTS_ERROR=0x12,
                    TTS_DISCONNECTED1=0xF,
                    TTS_INVALID=INVALID };
  
  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDBufferState { BUFFER_STATE_UNSET=0x0,
                        BUFFER_STATE_EMPTY=0x1,
                        BUFFER_STATE_PARTIAL_FULL=0x4,
                        BUFFER_STATE_FULL=0x8
                      };
  
  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDChannelStatus { CHANNEL_STATUS_LOCKED=0x20,
                          CHANNEL_STATUS_IN_SYNC=0x10,
                          CHANNEL_STATUS_APV1_ADDRESS_GOOD=0x08,
                          CHANNEL_STATUS_APV0_NO_ERROR_BIT=0x04,
			  CHANNEL_STATUS_APV0_ADDRESS_GOOD=0x02,
                          CHANNEL_STATUS_APV1_NO_ERROR_BIT=0x01,
			  CHANNEL_STATUS_NO_PROBLEMS=CHANNEL_STATUS_LOCKED|
			                             CHANNEL_STATUS_IN_SYNC|
                                                     CHANNEL_STATUS_APV1_ADDRESS_GOOD|
                                                     CHANNEL_STATUS_APV0_NO_ERROR_BIT|
                                                     CHANNEL_STATUS_APV0_ADDRESS_GOOD|
                                                     CHANNEL_STATUS_APV1_NO_ERROR_BIT
                        };

  //
  // Global function declarations
  //

  //used by these classes
  uint8_t internalFEDChannelNum(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum);
  void printHex(const void* pointer, const size_t length, std::ostream& os);
  //calculate the CRC for a FED buffer
  uint16_t calculateFEDBufferCRC(const uint8_t* buffer, const size_t lengthInBytes);
  //to make enums printable
  std::ostream& operator<<(std::ostream& os, const FEDBufferFormat& value);
  std::ostream& operator<<(std::ostream& os, const FEDHeaderType& value);
  std::ostream& operator<<(std::ostream& os, const FEDReadoutMode& value);
  std::ostream& operator<<(std::ostream& os, const FEDDataType& value);
  std::ostream& operator<<(std::ostream& os, const FEDDAQEventType& value);
  std::ostream& operator<<(std::ostream& os, const FEDTTSBits& value);
  std::ostream& operator<<(std::ostream& os, const FEDBufferState& value);
  std::ostream& operator<<(std::ostream& os, const FEDChannelStatus& value);
  //convert name of an element of enum to enum value (useful for getting values from config)
  FEDBufferFormat fedBufferFormatFromString(const std::string& bufferFormatString);
  FEDHeaderType fedHeaderTypeFromString(const std::string& headerTypeString);
  FEDReadoutMode fedReadoutModeFromString(const std::string& readoutModeString);
  FEDDataType fedDataTypeFromString(const std::string& dataTypeString);
  FEDDAQEventType fedDAQEventTypeFromString(const std::string& daqEventTypeString);

  //
  // Class definitions
  //
  
  //handles conversion between order of data in buffer in VR/PR modes (readout order) and strip order (physical order)
  class FEDStripOrdering
    {
    public:
      //convert strip/sample index in channel (ie 0-255) between physical and readout order
      static uint8_t physicalOrderForStripInChannel(const uint8_t readoutOrderStripIndexInChannel);
      static uint8_t readoutOrderForStripInChannel(const uint8_t physicalOrderStripIndexInChannel);
      //convert strip/sample index in APV (ie 0-127) between physical and readout order
      static uint8_t physicalOrderForStripInAPV(const uint8_t readoutOrderStripIndexInAPV);
      static uint8_t readoutOrderForStripInAPV(const uint8_t physicalOrderStripIndexInAPV);
    };

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQHeader
    {
    public:
      FEDDAQHeader() { }
      explicit FEDDAQHeader(const uint8_t* header);
      //0x5 in first fragment
      uint8_t boeNibble() const;
      uint8_t eventTypeNibble() const;
      FEDDAQEventType eventType() const;
      uint32_t l1ID() const;
      uint16_t bxID() const;
      uint16_t sourceID() const;
      uint8_t version() const;
      //0 if current header word is last, 1 otherwise
      bool hBit() const;
      bool lastHeader() const;
      void print(std::ostream& os) const;
      //used by digi2Raw
      const uint8_t* data() const;
      FEDDAQHeader& setEventType(const FEDDAQEventType evtType);
      FEDDAQHeader& setL1ID(const uint32_t l1ID);
      FEDDAQHeader& setBXID(const uint16_t bxID);
      FEDDAQHeader& setSourceID(const uint16_t sourceID);
      FEDDAQHeader(const uint32_t l1ID, const uint16_t bxID, const uint16_t sourceID, 
                   const FEDDAQEventType evtType = DAQ_EVENT_TYPE_PHYSICS);
    private:
      uint8_t header_[8];
    };

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQTrailer
    {
    public:
      FEDDAQTrailer() { }
      explicit FEDDAQTrailer(const uint8_t* trailer);
      //0xA in first fragment
      uint8_t eoeNibble() const;
      uint32_t eventLengthIn64BitWords() const;
      uint32_t eventLengthInBytes() const;
      uint16_t crc() const;
      //set to 1 if FRL detects a transmission error over S-link
      bool cBit() const;
      bool slinkTransmissionError() const { return cBit(); }
      //set to 1 if the FED ID is not the one expected by the FRL
      bool fBit() const;
      bool badSourceID() const { return fBit(); }
      uint8_t eventStatusNibble() const;
      uint8_t ttsNibble() const;
      FEDTTSBits ttsBits() const;
      //0 if the current trailer is the last, 1 otherwise
      bool tBit() const;
      bool lastTrailer() const { return !tBit(); }
      //set to 1 if the S-link sender card detects a CRC error (the CRC it computes is put in the CRC field)
      bool rBit() const;
      bool slinkCRCError() const { return rBit(); }
      void print(std::ostream& os) const;
      //used by digi2Raw
      const uint8_t* data() const;
      FEDDAQTrailer& setEventLengthIn64BitWords(const uint32_t eventLengthIn64BitWords);
      FEDDAQTrailer& setCRC(const uint16_t crc);
      FEDDAQTrailer& setSLinkTransmissionErrorBit(const bool bitSet);
      FEDDAQTrailer& setBadSourceIDBit(const bool bitSet);
      FEDDAQTrailer& setSLinkCRCErrorBit(const bool bitSet);
      FEDDAQTrailer& setEventStatusNibble(const uint8_t eventStatusNibble);
      FEDDAQTrailer& setTTSBits(const FEDTTSBits ttsBits);
      FEDDAQTrailer(const uint32_t eventLengthIn64BitWords, const uint16_t crc = 0, const FEDTTSBits ttsBits = TTS_READY,
                    const bool slinkTransmissionError = false, const bool badFEDID = false, const bool slinkCRCError = false,
                    const uint8_t eventStatusNibble = 0);
    private:
      uint8_t trailer_[8];
    };

  class FEDStatusRegister
    {
    public:
      FEDStatusRegister(const uint16_t fedStatusRegister);
      bool slinkFullFlag() const;
      bool trackerHeaderMonitorDataReadyFlag() const;
      bool qdrMemoryFullFlag() const;
      bool qdrMemoryPartialFullFlag() const;
      bool qdrMemoryEmptyFlag() const;
      bool l1aBxFIFOFullFlag() const;
      bool l1aBxFIFOPartialFullFlag() const;
      bool l1aBxFIFOEmptyFlag() const;
      FEDBufferState qdrMemoryState() const;
      FEDBufferState l1aBxFIFOState() const;
      bool feDataMissingFlag(const uint8_t internalFEUnitNum) const;
      void print(std::ostream& os) const;
      void printFlags(std::ostream& os) const;
      operator uint16_t () const;
      //used by digi2Raw
      FEDStatusRegister& setSLinkFullFlag(const bool bitSet);
      FEDStatusRegister& setTrackerHeaderMonitorDataReadyFlag(const bool bitSet);
      FEDStatusRegister& setQDRMemoryBufferState(const FEDBufferState state);
      FEDStatusRegister& setL1ABXFIFOBufferState(const FEDBufferState state);
      FEDStatusRegister(const FEDBufferState qdrMemoryBufferState = BUFFER_STATE_UNSET,
                        const FEDBufferState l1aBxFIFOBufferState = BUFFER_STATE_UNSET,
                        const bool trackerHeaderMonitorDataReadyFlagSet = false,
                        const bool slinkFullFlagSet = false);
    private:
      bool getBit(const uint8_t num) const;
      void setBit(const uint8_t num, const bool bitSet);
      void setQDRMemoryFullFlag(const bool bitSet);
      void setQDRMemoryPartialFullFlag(const bool bitSet);
      void setQDRMemoryEmptyFlag(const bool bitSet);
      void setL1ABXFIFOFullFlag(const bool bitSet);
      void setL1ABXFIFOPartialFullFlag(const bool bitSet);
      void setL1ABXFIFOEmptyFlag(const bool bitSet);
      uint16_t data_;
    };

  class TrackerSpecialHeader
    {
    public:
      TrackerSpecialHeader();
      //construct with a pointer to the data. The data will be coppied and swapped if necessary. 
      explicit TrackerSpecialHeader(const uint8_t* headerPointer);
      uint8_t bufferFormatByte() const;
      FEDBufferFormat bufferFormat() const;
      uint8_t headerTypeNibble() const;
      FEDHeaderType headerType() const;
      uint8_t trackerEventTypeNibble() const;
      FEDReadoutMode readoutMode() const;
      FEDDataType dataType() const;
      uint8_t apveAddress() const;
      uint8_t apvAddressErrorRegister() const;
      bool majorityAddressErrorForFEUnit(const uint8_t internalFEUnitNum) const;
      uint8_t feEnableRegister() const;
      bool feEnabled(const uint8_t internalFEUnitNum) const;
      uint8_t feOverflowRegister() const;
      bool feOverflow(const uint8_t internalFEUnitNum) const;
      uint16_t fedStatusRegisterWord() const;
      FEDStatusRegister fedStatusRegister() const;
      void print(std::ostream& os) const;
      //used by digi2Raw
      //returns ordered buffer (ie this may need to be swapped to get original order)
      const uint8_t* data() const;
      bool wasSwapped() const;
      TrackerSpecialHeader& setBufferFormat(const FEDBufferFormat newBufferFormat);
      TrackerSpecialHeader& setHeaderType(const FEDHeaderType headerType);
      TrackerSpecialHeader& setReadoutMode(const FEDReadoutMode readoutMode);
      TrackerSpecialHeader& setDataType(const FEDDataType dataType);
      TrackerSpecialHeader& setAPVEAddress(const uint8_t address);
      TrackerSpecialHeader& setAPVEAddressErrorRegister(const uint8_t addressErrorRegister);
      TrackerSpecialHeader& setAPVAddressErrorForFEUnit(const uint8_t internalFEUnitNum, const bool error);
      TrackerSpecialHeader& setFEEnableRegister(const uint8_t feEnableRegister);
      TrackerSpecialHeader& setFEEnableForFEUnit(const uint8_t internalFEUnitNum, const bool enabled);
      TrackerSpecialHeader& setFEOverflowRegister(const uint8_t feOverflowRegister);
      TrackerSpecialHeader& setFEOverflowForFEUnit(const uint8_t internalFEUnitNum, const bool overflow);
      TrackerSpecialHeader& setFEDStatusRegister(const FEDStatusRegister fedStatusRegister);
      TrackerSpecialHeader(const FEDBufferFormat bufferFormat, const FEDReadoutMode readoutMode,
                           const FEDHeaderType headerType, const FEDDataType dataType,
                           const uint8_t address = 0x00, const uint8_t addressErrorRegister = 0x00,
                           const uint8_t feEnableRegister = 0xFF, const uint8_t feOverflowRegister = 0x00,
                           const FEDStatusRegister fedStatusRegister = FEDStatusRegister());
    private:
      void setBufferFormatByte(const FEDBufferFormat newBufferFormat);
      void setHeaderTypeNibble(const uint8_t value);
      void setReadoutModeBits(const uint8_t value);
      void setDataTypeBit(const bool value);
      enum byteIndicies { FEDSTATUS=0, FEOVERFLOW=2, FEENABLE=3, ADDRESSERROR=4, APVEADDRESS=5, BUFFERTYPE=6, BUFFERFORMAT=7 };
      //copy of header, 32 bit word swapped if needed
      uint8_t specialHeader_[8];
      //was the header word swapped wrt order in buffer?
      bool wordSwapped_;
    };

  class FEDBackendStatusRegister
    {
    public:
      FEDBackendStatusRegister(const uint32_t backendStatusRegister);
      bool internalFreezeFlag() const;
      bool slinkDownFlag() const;
      bool slinkFullFlag() const;
      bool backpressureFlag() const;
      bool ttcReadyFlag() const;
      bool trackerHeaderMonitorDataReadyFlag() const;
      FEDBufferState qdrMemoryState() const;
      FEDBufferState frameAddressFIFOState() const;
      FEDBufferState totalLengthFIFOState() const;
      FEDBufferState trackerHeaderFIFOState() const;
      FEDBufferState l1aBxFIFOState() const;
      FEDBufferState feEventLengthFIFOState() const;
      FEDBufferState feFPGABufferState() const;
      void print(std::ostream& os) const;
      void printFlags(std::ostream& os) const;
      operator uint32_t () const;
      //used by digi2Raw
      FEDBackendStatusRegister& setInternalFreezeFlag(const bool bitSet);
      FEDBackendStatusRegister& setSLinkDownFlag(const bool bitSet);
      FEDBackendStatusRegister& setSLinkFullFlag(const bool bitSet);
      FEDBackendStatusRegister& setBackpressureFlag(const bool bitSet);
      FEDBackendStatusRegister& setTTCReadyFlag(const bool bitSet);
      FEDBackendStatusRegister& setTrackerHeaderMonitorDataReadyFlag(const bool bitSet);
      FEDBackendStatusRegister& setQDRMemoryState(const FEDBufferState state);
      FEDBackendStatusRegister& setFrameAddressFIFOState(const FEDBufferState state);
      FEDBackendStatusRegister& setTotalLengthFIFOState(const FEDBufferState state);
      FEDBackendStatusRegister& setTrackerHeaderFIFOState(const FEDBufferState state);
      FEDBackendStatusRegister& setL1ABXFIFOState(const FEDBufferState state);
      FEDBackendStatusRegister& setFEEventLengthFIFOState(const FEDBufferState state);
      FEDBackendStatusRegister& setFEFPGABufferState(const FEDBufferState state);
      FEDBackendStatusRegister(const FEDBufferState qdrMemoryBufferState = BUFFER_STATE_UNSET,
                               const FEDBufferState frameAddressFIFOBufferState = BUFFER_STATE_UNSET,
                               const FEDBufferState totalLengthFIFOBufferState = BUFFER_STATE_UNSET,
                               const FEDBufferState trackerHeaderFIFOBufferState = BUFFER_STATE_UNSET,
                               const FEDBufferState l1aBxFIFOBufferState = BUFFER_STATE_UNSET,
                               const FEDBufferState feEventLengthFIFOBufferState = BUFFER_STATE_UNSET,
                               const FEDBufferState feFPGABufferState = BUFFER_STATE_UNSET,
                               const bool backpressure = false, const bool slinkFull = false,
                               const bool slinkDown = false, const bool internalFreeze = false,
                               const bool trackerHeaderMonitorDataReady = false, const bool ttcReady = true);                               
    private:
      bool getBit(const uint8_t num) const;
      void setBit(const uint8_t num, const bool bitSet);
      //get the state of the buffer in position 'bufferPosition'
      FEDBufferState getBufferState(const uint8_t bufferPosition) const;
      //set the state of the buffer in position 'bufferPosition' to state 'state'
      void setBufferSate(const uint8_t bufferPosition, const FEDBufferState state);
      void printFlagsForBuffer(const FEDBufferState bufferState, const std::string name, std::ostream& os) const;
      //constants marking order of flags in buffer
      //eg. bit offset for L1A/BX FIFO Partial full flag is STATE_OFFSET_PARTIAL_FULL+BUFFER_POSITION_L1ABX_FIFO
      //    bit offset for total length FIFO empty flag is STATE_OFFSET_EMPTY+BUFFER_POSITION_TOTAL_LENGTH_FIFO
      //see BE FPGA technical description
      enum bufferPositions { BUFFER_POSITION_QDR_MEMORY=0,
                             BUFFER_POSITION_FRAME_ADDRESS_FIFO=1,
                             BUFFER_POSITION_TOTAL_LENGTH_FIFO=2,
                             BUFFER_POSITION_TRACKER_HEADER_FIFO=3,
                             BUFFER_POSITION_L1ABX_FIFO=4,
                             BUFFER_POSITION_FE_EVENT_LENGTH_FIFO=5,
                             BUFFER_POSITION_FE_FPGA_BUFFER=6 };
      enum stateOffsets { STATE_OFFSET_FULL=8,
                          STATE_OFFSET_PARTIAL_FULL=16,
                          STATE_OFFSET_EMPTY=24 };
      uint32_t data_;
    };

  class FEDFEHeader
    {
    public:
      //factory function: allocates new FEDFEHeader derrivative of appropriate type
      static std::auto_ptr<FEDFEHeader> newFEHeader(const FEDHeaderType headerType, const uint8_t* headerBuffer);
      //used by digi2Raw
      static std::auto_ptr<FEDFEHeader> newFEHeader(const FEDHeaderType headerType);
      //create a buffer to use with digi2Raw
      static std::auto_ptr<FEDFEHeader> newFEFakeHeader(const FEDHeaderType headerType);
      virtual ~FEDFEHeader();
      //the length of the header
      virtual size_t lengthInBytes() const = 0;
      //check that there are no errors indicated in which ever error bits are available in the header
      //check bits for both APVs on a channel
      bool checkChannelStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const;
      virtual bool checkChannelStatusBits(const uint8_t internalFEDChannelNum) const = 0;
      //check bits for one APV
      bool checkStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum) const;
      virtual bool checkStatusBits(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const = 0;
      virtual void print(std::ostream& os) const = 0;
      virtual FEDFEHeader* clone() const = 0;
      //used by digi2Raw
      virtual const uint8_t* data() const = 0;
      virtual void setChannelStatus(const uint8_t internalFEDChannelNum, const FEDChannelStatus status) = 0;
      virtual void setFEUnitMajorityAddress(const uint8_t internalFEUnitNum, const uint8_t address) = 0;
      virtual void setBEStatusRegister(const FEDBackendStatusRegister beStatusRegister) = 0;
      virtual void setFEUnitLength(const uint8_t internalFEUnitNum, const uint16_t length) = 0;
      void setChannelStatus(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const FEDChannelStatus status);
    };

  class FEDAPVErrorHeader : public FEDFEHeader
    {
    public:
      explicit FEDAPVErrorHeader(const uint8_t* headerBuffer);
      virtual ~FEDAPVErrorHeader();
      virtual size_t lengthInBytes() const;
      virtual bool checkChannelStatusBits(const uint8_t internalFEDChannelNum) const;
      virtual bool checkStatusBits(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const;
      virtual void print(std::ostream& os) const;
      virtual FEDAPVErrorHeader* clone() const;
      //used by digi2Raw
      virtual const uint8_t* data() const;
      FEDAPVErrorHeader& setAPVStatusBit(const uint8_t internalFEDChannelNum, const uint8_t apvNum, const bool apvGood);
      FEDAPVErrorHeader& setAPVStatusBit(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum, const bool apvGood);
      FEDAPVErrorHeader(const std::vector<bool>& apvsGood = std::vector<bool>(APVS_PER_FED,true));
      //Information which is not present in APVError mode is allowed to be set here so that the methods can be called on the base class without caring
      //if the values need to be set.
      virtual void setChannelStatus(const uint8_t internalFEDChannelNum, const FEDChannelStatus status);
      virtual void setFEUnitMajorityAddress(const uint8_t internalFEUnitNum, const uint8_t address);
      virtual void setBEStatusRegister(const FEDBackendStatusRegister beStatusRegister);
      virtual void setFEUnitLength(const uint8_t internalFEUnitNum, const uint16_t length);
    private:
      static const size_t APV_ERROR_HEADER_SIZE_IN_64BIT_WORDS = 3;
      static const size_t APV_ERROR_HEADER_SIZE_IN_BYTES = APV_ERROR_HEADER_SIZE_IN_64BIT_WORDS*8;
      uint8_t header_[APV_ERROR_HEADER_SIZE_IN_BYTES];
    };

  class FEDFullDebugHeader : public FEDFEHeader
    {
    public:
      explicit FEDFullDebugHeader(const uint8_t* headerBuffer);
      virtual ~FEDFullDebugHeader();
      virtual size_t lengthInBytes() const;
      virtual bool checkChannelStatusBits(const uint8_t internalFEDChannelNum) const;
      virtual bool checkStatusBits(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const;
      virtual void print(std::ostream& os) const;
      virtual FEDFullDebugHeader* clone() const;
      
      uint8_t feUnitMajorityAddress(const uint8_t internalFEUnitNum) const;
      FEDBackendStatusRegister beStatusRegister() const;
      uint32_t daqRegister() const;
      uint32_t daqRegister2() const;
      uint16_t feUnitLength(const uint8_t internalFEUnitNum) const;
      bool fePresent(const uint8_t internalFEUnitNum) const;
      
      FEDChannelStatus getChannelStatus(const uint8_t internalFEDChannelNum) const;
      FEDChannelStatus getChannelStatus(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const;
      
      //These methods return true if there was an error of the appropriate type (ie if the error bit is 0).
      //They return false if the error could not occur due to a more general error.
      //was channel unlocked
      bool unlocked(const uint8_t internalFEDChannelNum) const;
      bool unlocked(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const;
      //was channel out of sync if it was unlocked
      bool outOfSync(const uint8_t internalFEDChannelNum) const;
      bool outOfSync(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const;
      //was there an internal APV error if it was in sync
      bool apvError(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const;
      bool apvError(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum) const;
      //was the APV address wrong if it was in sync (does not depend on APV internal error bit)
      bool apvAddressError(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const;
      bool apvAddressError(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum) const;
      
      //used by digi2Raw
      virtual const uint8_t* data() const;
      virtual void setChannelStatus(const uint8_t internalFEDChannelNum, const FEDChannelStatus status);
      virtual void setFEUnitMajorityAddress(const uint8_t internalFEUnitNum, const uint8_t address);
      virtual void setBEStatusRegister(const FEDBackendStatusRegister beStatusRegister);
      virtual void setDAQRegister(const uint32_t daqRegister);
      virtual void setDAQRegister2(const uint32_t daqRegister2);
      virtual void setFEUnitLength(const uint8_t internalFEUnitNum, const uint16_t length);
      FEDFullDebugHeader(const std::vector<uint16_t>& feUnitLengths = std::vector<uint16_t>(FEUNITS_PER_FED,0),
                         const std::vector<uint8_t>& feMajorityAddresses = std::vector<uint8_t>(FEUNITS_PER_FED,0),
                         const std::vector<FEDChannelStatus>& channelStatus = std::vector<FEDChannelStatus>(FEDCH_PER_FED,CHANNEL_STATUS_NO_PROBLEMS),
                         const FEDBackendStatusRegister beStatusRegister = FEDBackendStatusRegister(),
                         const uint32_t daqRegister = 0, const uint32_t daqRegister2 = 0);
    private:
      bool getBit(const uint8_t internalFEDChannelNum, const uint8_t bit) const;
      static uint32_t get32BitWordFrom(const uint8_t* startOfWord);
      static void set32BitWordAt(uint8_t* startOfWord, const uint32_t value);
      const uint8_t* feWord(const uint8_t internalFEUnitNum) const;
      uint8_t* feWord(const uint8_t internalFEUnitNum);
      void setBit(const uint8_t internalFEDChannelNum, const uint8_t bit, const bool value);
      
      //These methods return true if there was an error of the appropriate type (ie if the error bit is 0).
      //They ignore any previous errors which make the status bits meaningless and return the value of the bit anyway.
      //In general, the methods above which only return an error for the likely cause are more useful.
      bool unlockedFromBit(const uint8_t internalFEDChannelNum) const;
      bool outOfSyncFromBit(const uint8_t internalFEDChannelNum) const;
      bool apvErrorFromBit(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const;
      bool apvAddressErrorFromBit(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const;
      
      //following methods set the bits to 1 (no error) if value is false
      void setUnlocked(const uint8_t internalFEDChannelNum, const bool value);
      void setOutOfSync(const uint8_t internalFEDChannelNum, const bool value);
      void setAPVAddressError(const uint8_t internalFEDChannelNum, const uint8_t apvNum, const bool value);
      void setAPVError(const uint8_t internalFEDChannelNum, const uint8_t apvNum, const bool value);
      static const size_t FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS = FEUNITS_PER_FED*2;
      static const size_t FULL_DEBUG_HEADER_SIZE_IN_BYTES = FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS*8;
      uint8_t header_[FULL_DEBUG_HEADER_SIZE_IN_BYTES];
    };

  //holds information about position of a channel in the buffer for use by unpacker
  class FEDChannel
    {
    public:
      FEDChannel(const uint8_t*const data, const size_t offset, const uint16_t length);
      //gets length from first 2 bytes (assuming normal FED channel)
      FEDChannel(const uint8_t*const data, const size_t offset);
      uint16_t length() const;
      const uint8_t* data() const;
      size_t offset() const;
      uint16_t cmMedian(const uint8_t apvIndex) const;
    private:
      friend class FEDBuffer;
      //third byte of channel data for normal FED channels
      uint8_t packetCode() const;
      const uint8_t* data_;
      size_t offset_;
      uint16_t length_;
    };

  //base class for sistrip FED buffers which have a DAQ header/trailer and tracker special header
  class FEDBufferBase
    {
    public:
      FEDBufferBase(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowUnrecognizedFormat = false);
      virtual ~FEDBufferBase();
      //dump buffer to stream
      void dump(std::ostream& os) const;
      //dump original buffer before word swapping
      void dumpOriginalBuffer(std::ostream& os) const;
      virtual void print(std::ostream& os) const;
      //calculate the CRC from the buffer
      uint16_t calcCRC() const;
  
      //methods to get parts of the buffer
      FEDDAQHeader daqHeader() const;
      FEDDAQTrailer daqTrailer() const;
      size_t bufferSize() const;
      TrackerSpecialHeader trackerSpecialHeader() const;
      //methods to get info from DAQ header
      FEDDAQEventType daqEventType() const;
      uint32_t daqLvl1ID() const;
      uint16_t daqBXID() const;
      uint16_t daqSourceID() const;
      uint16_t sourceID() const;
      //methods to get info from DAQ trailer
      uint32_t daqEventLengthIn64bitWords() const;
      uint32_t daqEventLengthInBytes() const;
      uint16_t daqCRC() const;
      FEDTTSBits daqTTSState() const;
      //methods to get info from the tracker special header
      FEDBufferFormat bufferFormat() const;
      FEDHeaderType headerType() const;
      FEDReadoutMode readoutMode() const;
      FEDDataType dataType() const;
      uint8_t apveAddress() const;
      bool majorityAddressErrorForFEUnit(const uint8_t internalFEUnitNum) const;
      bool feEnabled(const uint8_t internalFEUnitNum) const;
      uint8_t nFEUnitsEnabled() const;
      bool feOverflow(const uint8_t internalFEUnitNum) const;
      FEDStatusRegister fedStatusRegister() const;
      
      //check that channel has no errors
      virtual bool channelGood(const uint8_t internalFEDChannelNum) const;
      bool channelGood(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const;
      //return channel object for channel
      const FEDChannel& channel(const uint8_t internalFEDChannelNum) const;
      const FEDChannel& channel(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const;
  
      //summary checks
      //check that tracker special header is valid (does not check for FE unit errors indicated in special header)
      bool doTrackerSpecialHeaderChecks() const;
      //check for errors in DAQ heaqder and trailer (not including bad CRC)
      bool doDAQHeaderAndTrailerChecks() const;
      //do both
      virtual bool doChecks() const;
      //print the result of all detailed checks
      virtual std::string checkSummary() const;
  
      //detailed checks
      bool checkCRC() const;
      bool checkMajorityAddresses() const;
      //methods to check tracker special header
      bool checkBufferFormat() const;
      bool checkHeaderType() const;
      bool checkReadoutMode() const;
      bool checkAPVEAddressValid() const;
      bool checkNoFEOverflows() const;
      //methods to check daq header and trailer
      bool checkNoSlinkCRCError() const;
      bool checkNoSLinkTransmissionError() const;
      bool checkSourceIDs() const;
      bool checkNoUnexpectedSourceID() const;
      bool checkNoExtraHeadersOrTrailers() const;
      bool checkLengthFromTrailer() const;
    protected:
      const uint8_t* getPointerToDataAfterTrackerSpecialHeader() const;
      const uint8_t* getPointerToByteAfterEndOfPayload() const;
      FEDBufferBase(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowUnrecognizedFormat, const bool fillChannelVector);
      std::vector<FEDChannel> channels_;
    private:
      void init(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowUnrecognizedFormat);
      const uint8_t* originalBuffer_;
      const uint8_t* orderedBuffer_;
      const size_t bufferSize_;
      FEDDAQHeader daqHeader_;
      FEDDAQTrailer daqTrailer_;
      TrackerSpecialHeader specialHeader_;
    };

  //
  // Inline function definitions
  //
  
  inline std::ostream& operator << (std::ostream& os, const FEDBufferBase& obj) { obj.print(os); os << obj.checkSummary(); return os; }

  inline uint8_t internalFEDChannelNum(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum)
    {
      return (internalFEUnitNum*FEDCH_PER_FEUNIT + internalFEUnitChannelNum);
    }

  inline std::ostream& operator << (std::ostream& os, const FEDDAQHeader& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDDAQTrailer& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const TrackerSpecialHeader& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDStatusRegister& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDFEHeader& obj) { obj.print(os); return os; }

  //FEDStripOrdering

  inline uint8_t FEDStripOrdering::physicalOrderForStripInChannel(const uint8_t readoutOrderStripIndexInChannel)
    {
      return physicalOrderForStripInAPV(readoutOrderStripIndexInChannel/2) + (readoutOrderStripIndexInChannel%2)*STRIPS_PER_APV;
    }
  
  inline uint8_t FEDStripOrdering::readoutOrderForStripInChannel(const uint8_t physicalOrderStripIndexInChannel)
    {
      return ( readoutOrderForStripInAPV(physicalOrderStripIndexInChannel%128)*2 + (physicalOrderStripIndexInChannel/128) );
    }
  
  inline uint8_t FEDStripOrdering::physicalOrderForStripInAPV(const uint8_t readout_order)
    {
      return ( (32 * (readout_order%4)) +
               (8 * static_cast<uint16_t>(static_cast<float>(readout_order)/4.0)) -
               (31 * static_cast<uint16_t>(static_cast<float>(readout_order)/16.0))
             );
    }
  
  inline uint8_t FEDStripOrdering::readoutOrderForStripInAPV(const uint8_t physical_order)
    {
      return ( 4*((static_cast<uint16_t>((static_cast<float>(physical_order)/8.0)))%4) +
               static_cast<uint16_t>(static_cast<float>(physical_order)/32.0) +
               16*(physical_order%8)
             );
    }

  //TrackerSpecialHeader

  inline TrackerSpecialHeader::TrackerSpecialHeader()
    : wordSwapped_(false)
    {
    }
  
  inline uint8_t TrackerSpecialHeader::bufferFormatByte() const
    { return specialHeader_[BUFFERFORMAT]; }
  
  inline uint8_t TrackerSpecialHeader::headerTypeNibble() const
    { return ( (specialHeader_[BUFFERTYPE] & 0xF0) >> 4 ); }
  
  inline uint8_t TrackerSpecialHeader::trackerEventTypeNibble() const
    { return (specialHeader_[BUFFERTYPE] & 0x0F); }
  
  inline uint8_t TrackerSpecialHeader::apveAddress() const
    { return specialHeader_[APVEADDRESS]; }
  
  inline uint8_t TrackerSpecialHeader::apvAddressErrorRegister() const
    { return specialHeader_[ADDRESSERROR]; }
  
  inline bool TrackerSpecialHeader::majorityAddressErrorForFEUnit(const uint8_t internalFEUnitNum) const
    {
      return ( !(readoutMode() == READOUT_MODE_SCOPE) && !( (0x1<<internalFEUnitNum) & apvAddressErrorRegister() ) );
    }
  
  inline uint8_t TrackerSpecialHeader::feEnableRegister() const
    { return specialHeader_[FEENABLE]; }
  
  inline bool TrackerSpecialHeader::feEnabled(const uint8_t internalFEUnitNum) const
    {
      return ( (0x1<<internalFEUnitNum) & feEnableRegister() );
    }
  
  inline uint8_t TrackerSpecialHeader::feOverflowRegister() const
    { return specialHeader_[FEOVERFLOW]; }
  
  inline bool TrackerSpecialHeader::feOverflow(const uint8_t internalFEUnitNum) const
    {
      return ( (0x1<<internalFEUnitNum) & feOverflowRegister() );
    }
  
  inline uint16_t TrackerSpecialHeader::fedStatusRegisterWord() const
    {
      //get 16 bits
      uint16_t statusRegister = ( (specialHeader_[(FEDSTATUS+1)]<<8) | specialHeader_[FEDSTATUS]);
      return statusRegister;
    }
  
  inline FEDStatusRegister TrackerSpecialHeader::fedStatusRegister() const
    { return FEDStatusRegister(fedStatusRegisterWord()); }
  
  inline void TrackerSpecialHeader::print(std::ostream& os) const
    { printHex(specialHeader_,8,os); }
  
  inline const uint8_t* TrackerSpecialHeader::data() const
    {
      return specialHeader_;
    }
  
  inline bool TrackerSpecialHeader::wasSwapped() const
    {
      return wordSwapped_;
    }
  
  inline void TrackerSpecialHeader::setHeaderTypeNibble(const uint8_t value)
    {
      specialHeader_[BUFFERTYPE] = ( (specialHeader_[BUFFERTYPE] & 0x0F) | ((value<<4) & 0xF0) );
    }
  
  inline void TrackerSpecialHeader::setReadoutModeBits(const uint8_t value)
    {
      specialHeader_[BUFFERTYPE] = ( (specialHeader_[BUFFERTYPE] & (~0x0E)) | (value & 0x0E) );
    }
      
  inline void TrackerSpecialHeader::setDataTypeBit(const bool value)
    {
      specialHeader_[BUFFERTYPE] = ( (specialHeader_[BUFFERTYPE] & (~0x01)) | (value ? 0x01 : 0x00) );
    }
  
  inline TrackerSpecialHeader& TrackerSpecialHeader::setAPVEAddress(const uint8_t address)
    {
      specialHeader_[APVEADDRESS] = address;
      return *this;
    }
  
  inline TrackerSpecialHeader& TrackerSpecialHeader::setAPVEAddressErrorRegister(const uint8_t addressErrorRegister)
    {
      specialHeader_[ADDRESSERROR] = addressErrorRegister;
      return *this;
    }
  
  inline TrackerSpecialHeader& TrackerSpecialHeader::setFEEnableRegister(const uint8_t feEnableRegister)
    {
      specialHeader_[FEENABLE] = feEnableRegister;
      return *this;
    }
  
  inline TrackerSpecialHeader& TrackerSpecialHeader::setFEOverflowRegister(const uint8_t feOverflowRegister)
    {
      specialHeader_[FEOVERFLOW] = feOverflowRegister;
      return *this;
    }
  
  inline TrackerSpecialHeader& TrackerSpecialHeader::setFEDStatusRegister(const FEDStatusRegister fedStatusRegister)
    {
      specialHeader_[FEDSTATUS] = (static_cast<uint16_t>(fedStatusRegister) & 0x00FF);
      specialHeader_[FEDSTATUS+1] = ( (static_cast<uint16_t>(fedStatusRegister) & 0xFF00) >> 8);
      return *this;
    }

  //FEDStatusRegister

  inline FEDStatusRegister::FEDStatusRegister(const uint16_t fedStatusRegister)
    : data_(fedStatusRegister) { }
  
  inline FEDStatusRegister::operator uint16_t () const
    { return data_; }
  
  inline bool FEDStatusRegister::getBit(const uint8_t num) const
    { return ( (0x1<<num) & (data_) ); }
  
  inline bool FEDStatusRegister::slinkFullFlag() const
    { return getBit(0); }
  
  inline bool FEDStatusRegister::trackerHeaderMonitorDataReadyFlag() const
    { return getBit(1); }
  
  inline bool FEDStatusRegister::qdrMemoryFullFlag() const
    { return getBit(2); }
  
  inline bool FEDStatusRegister::qdrMemoryPartialFullFlag() const
    { return getBit(3); }
  
  inline bool FEDStatusRegister::qdrMemoryEmptyFlag() const
    { return getBit(4); }
  
  inline bool FEDStatusRegister::l1aBxFIFOFullFlag() const
    { return getBit(5); }
  
  inline bool FEDStatusRegister::l1aBxFIFOPartialFullFlag() const
    { return getBit(6); }
  
  inline bool FEDStatusRegister::l1aBxFIFOEmptyFlag() const
    { return getBit(7); }
  
  inline bool FEDStatusRegister::feDataMissingFlag(const uint8_t internalFEUnitNum) const
    {
      return getBit(8+internalFEUnitNum);
    }
  
  inline void FEDStatusRegister::print(std::ostream& os) const
    { printHex(&data_,2,os); }
  
  inline FEDStatusRegister& FEDStatusRegister::setSLinkFullFlag(const bool bitSet)
    { setBit(0,bitSet); return *this; }
  
  inline FEDStatusRegister& FEDStatusRegister::setTrackerHeaderMonitorDataReadyFlag(const bool bitSet)
    { setBit(1,bitSet); return *this; }
  
  inline void FEDStatusRegister::setQDRMemoryFullFlag(const bool bitSet)
    { setBit(2,bitSet); }
  
  inline void FEDStatusRegister::setQDRMemoryPartialFullFlag(const bool bitSet)
    { setBit(3,bitSet); }
  
  inline void FEDStatusRegister::setQDRMemoryEmptyFlag(const bool bitSet)
    { setBit(4,bitSet); }
  
  inline void FEDStatusRegister::setL1ABXFIFOFullFlag(const bool bitSet)
    { setBit(5,bitSet); }
  
  inline void FEDStatusRegister::setL1ABXFIFOPartialFullFlag(const bool bitSet)
    { setBit(6,bitSet); }
  
  inline void FEDStatusRegister::setL1ABXFIFOEmptyFlag(const bool bitSet)
    { setBit(7,bitSet); }
  
  inline FEDStatusRegister::FEDStatusRegister(const FEDBufferState qdrMemoryBufferState, const FEDBufferState l1aBxFIFOBufferState,
                                              const bool trackerHeaderMonitorDataReadyFlagSet, const bool slinkFullFlagSet)
    : data_(0x0000)
    {
      setSLinkFullFlag(slinkFullFlagSet);
      setTrackerHeaderMonitorDataReadyFlag(trackerHeaderMonitorDataReadyFlagSet);
      setQDRMemoryBufferState(qdrMemoryBufferState);
      setL1ABXFIFOBufferState(l1aBxFIFOBufferState);
    }

  //FEDBackendStatusRegister

  inline FEDBackendStatusRegister::FEDBackendStatusRegister(const uint32_t backendStatusRegister)
    : data_(backendStatusRegister) { }
  
  inline FEDBackendStatusRegister::operator uint32_t () const
    { return data_; }
  
  inline void FEDBackendStatusRegister::print(std::ostream& os) const
    { printHex(&data_,4,os); }
  
  inline bool FEDBackendStatusRegister::getBit(const uint8_t num) const
    { return ( (0x1<<num) & (data_) ); }  
  
  inline bool FEDBackendStatusRegister::internalFreezeFlag() const
    { return getBit(1); }
  
  inline bool FEDBackendStatusRegister::slinkDownFlag() const
    { return getBit(2); }
  
  inline bool FEDBackendStatusRegister::slinkFullFlag() const
    { return getBit(3); }
  
  inline bool FEDBackendStatusRegister::backpressureFlag() const
    { return getBit(4); }
  
  inline bool FEDBackendStatusRegister::ttcReadyFlag() const
    { return getBit(6); }
  
  inline bool FEDBackendStatusRegister::trackerHeaderMonitorDataReadyFlag() const
    { return getBit(7); }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setInternalFreezeFlag(const bool bitSet)
    { setBit(1,bitSet); return *this; }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setSLinkDownFlag(const bool bitSet)
    { setBit(2,bitSet); return *this; }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setSLinkFullFlag(const bool bitSet)
    { setBit(3,bitSet); return *this; }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setBackpressureFlag(const bool bitSet)
    { setBit(4,bitSet); return *this; }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setTTCReadyFlag(const bool bitSet)
    { setBit(6,bitSet); return *this; }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setTrackerHeaderMonitorDataReadyFlag(const bool bitSet)
    { setBit(7,bitSet); return *this; }
  
  inline FEDBufferState FEDBackendStatusRegister::qdrMemoryState() const
    {
      return getBufferState(BUFFER_POSITION_QDR_MEMORY);
    }
  
  inline FEDBufferState FEDBackendStatusRegister::frameAddressFIFOState() const
    {
      return getBufferState(BUFFER_POSITION_FRAME_ADDRESS_FIFO);
    }
  
  inline FEDBufferState FEDBackendStatusRegister::totalLengthFIFOState() const
    {
      return getBufferState(BUFFER_POSITION_TOTAL_LENGTH_FIFO);
    }
  
  inline FEDBufferState FEDBackendStatusRegister::trackerHeaderFIFOState() const
    {
      return getBufferState(BUFFER_POSITION_TRACKER_HEADER_FIFO);
    }
  
  inline FEDBufferState FEDBackendStatusRegister::l1aBxFIFOState() const
    {
      return getBufferState(BUFFER_POSITION_L1ABX_FIFO);
    }
  
  inline FEDBufferState FEDBackendStatusRegister::feEventLengthFIFOState() const
    {
      return getBufferState(BUFFER_POSITION_FE_EVENT_LENGTH_FIFO);
    }
  
  inline FEDBufferState FEDBackendStatusRegister::feFPGABufferState() const
    {
      return getBufferState(BUFFER_POSITION_FE_FPGA_BUFFER);
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setQDRMemoryState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_QDR_MEMORY,state);
      return *this;
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setFrameAddressFIFOState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_FRAME_ADDRESS_FIFO,state);
      return *this;
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setTotalLengthFIFOState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_TOTAL_LENGTH_FIFO,state);
      return *this;
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setTrackerHeaderFIFOState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_TRACKER_HEADER_FIFO,state);
      return *this;
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setL1ABXFIFOState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_L1ABX_FIFO,state);
      return *this;
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setFEEventLengthFIFOState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_FE_EVENT_LENGTH_FIFO,state);
      return *this;
    }
  
  inline FEDBackendStatusRegister& FEDBackendStatusRegister::setFEFPGABufferState(const FEDBufferState state)
    {
      setBufferSate(BUFFER_POSITION_FE_FPGA_BUFFER,state);
      return *this;
    }

  //FEDFEHeader

  inline std::auto_ptr<FEDFEHeader> FEDFEHeader::newFEHeader(const FEDHeaderType headerType, const uint8_t* headerBuffer)
    {
      switch (headerType) {
      case HEADER_TYPE_FULL_DEBUG:
        return std::auto_ptr<FEDFEHeader>(new FEDFullDebugHeader(headerBuffer));
      case HEADER_TYPE_APV_ERROR:
        return std::auto_ptr<FEDFEHeader>(new FEDAPVErrorHeader(headerBuffer));
      default:
        return std::auto_ptr<FEDFEHeader>();
      }
    }
  
  inline std::auto_ptr<FEDFEHeader> FEDFEHeader::newFEHeader(const FEDHeaderType headerType)
    {
      switch (headerType) {
      case HEADER_TYPE_FULL_DEBUG:
        return std::auto_ptr<FEDFEHeader>(new FEDFullDebugHeader());
      case HEADER_TYPE_APV_ERROR:
        return std::auto_ptr<FEDFEHeader>(new FEDAPVErrorHeader());
      default:
        return std::auto_ptr<FEDFEHeader>();
      }
    }
  
  inline std::auto_ptr<FEDFEHeader> FEDFEHeader::newFEFakeHeader(const FEDHeaderType headerType)
    {
      switch (headerType) {
      case HEADER_TYPE_FULL_DEBUG:
        return std::auto_ptr<FEDFEHeader>(new FEDFullDebugHeader);
      case HEADER_TYPE_APV_ERROR:
        return std::auto_ptr<FEDFEHeader>(new FEDAPVErrorHeader);
      default:
        return std::auto_ptr<FEDFEHeader>();
      }
    }
  
  inline bool FEDFEHeader::checkChannelStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const
    {
      return checkChannelStatusBits(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum));
    }
  
  inline bool FEDFEHeader::checkStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum) const
    {
      return checkStatusBits(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum),apvNum);
    }
  
  inline void FEDFEHeader::setChannelStatus(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const FEDChannelStatus status)
    {
      this->setChannelStatus(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum),status);
    }
  
  inline FEDAPVErrorHeader::FEDAPVErrorHeader(const uint8_t* headerBuffer)
    {
      memcpy(header_,headerBuffer,APV_ERROR_HEADER_SIZE_IN_BYTES);
    }
  
  inline FEDAPVErrorHeader& FEDAPVErrorHeader::setAPVStatusBit(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum,
                                                               const uint8_t apvNum, const bool apvGood)
    {
      return setAPVStatusBit(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum),apvNum,apvGood);
    }
  
  inline FEDFullDebugHeader::FEDFullDebugHeader(const uint8_t* headerBuffer)
    {
      memcpy(header_,headerBuffer,FULL_DEBUG_HEADER_SIZE_IN_BYTES);
    }
  
  inline uint8_t FEDFullDebugHeader::feUnitMajorityAddress(const uint8_t internalFEUnitNum) const
    {
      return feWord(internalFEUnitNum)[9];
    }
  
  inline FEDBackendStatusRegister FEDFullDebugHeader::beStatusRegister() const
    {
      return FEDBackendStatusRegister(get32BitWordFrom(feWord(0)+10));
    }
  
  inline uint32_t FEDFullDebugHeader::daqRegister() const
    {
      return get32BitWordFrom(feWord(7)+10);
    }
  
  inline uint32_t FEDFullDebugHeader::daqRegister2() const
    {
      return get32BitWordFrom(feWord(6)+10);
    }
  
  inline uint16_t FEDFullDebugHeader::feUnitLength(const uint8_t internalFEUnitNum) const
    {
      return ( (feWord(internalFEUnitNum)[15]<<8) | (feWord(internalFEUnitNum)[14]) );
    }
  
  inline bool FEDFullDebugHeader::fePresent(const uint8_t internalFEUnitNum) const
    {
      return (feUnitLength(internalFEUnitNum) != 0);
    }
  
  inline bool FEDFullDebugHeader::unlocked(const uint8_t internalFEDChannelNum) const
    {
      return unlockedFromBit(internalFEDChannelNum);
    }
  
  inline bool FEDFullDebugHeader::unlocked(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const
    {
      return unlocked(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum));
    }
  
  inline bool FEDFullDebugHeader::outOfSync(const uint8_t internalFEDChannelNum) const
    {
      return ( !unlocked(internalFEDChannelNum) && outOfSyncFromBit(internalFEDChannelNum) );
    }
  
  inline bool FEDFullDebugHeader::outOfSync(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const
    {
      return outOfSync(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum));
    }
  
  inline bool FEDFullDebugHeader::apvError(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const
    {
      return ( !unlockedFromBit(internalFEDChannelNum) &&
               !outOfSyncFromBit(internalFEDChannelNum) &&
               apvErrorFromBit(internalFEDChannelNum,apvNum) );
    }
  
  inline bool FEDFullDebugHeader::apvError(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum) const
    {
      return apvError(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum),apvNum);
    }
  
  inline bool FEDFullDebugHeader::apvAddressError(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const
    {
      return ( !unlockedFromBit(internalFEDChannelNum) &&
               !outOfSyncFromBit(internalFEDChannelNum) &&
               apvAddressErrorFromBit(internalFEDChannelNum,apvNum) );
    }
  
  inline bool FEDFullDebugHeader::apvAddressError(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum, const uint8_t apvNum) const
    {
      return apvAddressError(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum),apvNum);
    }
  
  inline FEDChannelStatus FEDFullDebugHeader::getChannelStatus(const uint8_t internalFEUnitNum, const uint8_t internalFEUnitChannelNum) const
    {
      return getChannelStatus(internalFEDChannelNum(internalFEUnitNum,internalFEUnitChannelNum));
    }
  
  inline bool FEDFullDebugHeader::unlockedFromBit(const uint8_t internalFEDChannelNum) const
    {
      return !getBit(internalFEDChannelNum,5);
    }
  
  inline bool FEDFullDebugHeader::outOfSyncFromBit(const uint8_t internalFEDChannelNum) const
    {
      return !getBit(internalFEDChannelNum,4);
    }
  
  inline bool FEDFullDebugHeader::apvErrorFromBit(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const
    {
      //Discovered March 2012: two bits inverted in firmware. Decided
      //to update documentation but keep firmware identical for
      //backward compatibility. So status bit order is actually:
      //apvErr1 - apvAddrErr0 - apvErr0 - apvAddrErr1 - OOS - unlocked.
      //Before, it was: return !getBit(internalFEDChannelNum,0+2*apvNum);

      return !getBit(internalFEDChannelNum,0+2*(1-apvNum));
    }
  
  inline bool FEDFullDebugHeader::apvAddressErrorFromBit(const uint8_t internalFEDChannelNum, const uint8_t apvNum) const
    {
      return !getBit(internalFEDChannelNum,1+2*apvNum);
    }
  
  inline bool FEDFullDebugHeader::getBit(const uint8_t internalFEDChannelNum, const uint8_t bit) const
    {
      const uint8_t* pFEWord = feWord(internalFEDChannelNum / FEDCH_PER_FEUNIT);
      const uint8_t bitInFeWord = ((FEDCH_PER_FEUNIT-1) - (internalFEDChannelNum%FEDCH_PER_FEUNIT)) * 6 + bit;
      return ( pFEWord[bitInFeWord/8] & (0x1 << (bitInFeWord%8)) );
    }
  
  inline uint32_t FEDFullDebugHeader::get32BitWordFrom(const uint8_t* startOfWord)
    {
      return ( startOfWord[0] | (startOfWord[1]<<8) | (startOfWord[2]<<16) | (startOfWord[3]<<24) );
    }
  
  inline void FEDFullDebugHeader::set32BitWordAt(uint8_t* startOfWord, const uint32_t value)
    {
      memcpy(startOfWord,&value,4);
    }
  
  inline const uint8_t* FEDFullDebugHeader::feWord(const uint8_t internalFEUnitNum) const
    {
      return header_+internalFEUnitNum*2*8;
    }
  
  //re-use const method
  inline uint8_t* FEDFullDebugHeader::feWord(const uint8_t internalFEUnitNum)
    {
      return const_cast<uint8_t*>(static_cast<const FEDFullDebugHeader*>(this)->feWord(internalFEUnitNum));
    }
  
  inline void FEDFullDebugHeader::setUnlocked(const uint8_t internalFEDChannelNum, const bool value)
    {
      setBit(internalFEDChannelNum,5,!value);
    }
  
  inline void FEDFullDebugHeader::setOutOfSync(const uint8_t internalFEDChannelNum, const bool value)
    {
      setBit(internalFEDChannelNum,4,!value);
    }
  
  inline void FEDFullDebugHeader::setAPVAddressError(const uint8_t internalFEDChannelNum, const uint8_t apvNum, const bool value)
    {
      setBit(internalFEDChannelNum,1+2*apvNum,!value);
    }
  
  inline void FEDFullDebugHeader::setAPVError(const uint8_t internalFEDChannelNum, const uint8_t apvNum, const bool value)
    {
      //Discovered March 2012: two bits inverted in firmware. Decided
      //to update documentation but keep firmware identical for
      //backward compatibility. So status bit order is actually:
      //apvErr1 - apvAddrErr0 - apvErr0 - apvAddrErr1 - OOS - unlocked.
      //Before, it was: return !getBit(internalFEDChannelNum,0+2*apvNum);

      setBit(internalFEDChannelNum,0+2*(1-apvNum),!value);
    }

  //FEDDAQHeader

  inline FEDDAQHeader::FEDDAQHeader(const uint8_t* header)
    {
      memcpy(header_,header,8);
    }
  
  inline uint8_t FEDDAQHeader::boeNibble() const
    {
      return ( (header_[7] & 0xF0) >> 4 );
    }
  
  inline uint8_t FEDDAQHeader::eventTypeNibble() const
    {
      return (header_[7] & 0x0F);
    }
  
  inline uint32_t FEDDAQHeader::l1ID() const
    {
      return ( header_[4] | (header_[5]<<8) | (header_[6]<<16) );
    }
  
  inline uint16_t FEDDAQHeader::bxID() const
    {
      return ( (header_[3]<<4) | ((header_[2]&0xF0)>>4) );
    }
  
  inline uint16_t FEDDAQHeader::sourceID() const
    {
      return ( ((header_[2]&0x0F)<<8) | header_[1] );
    }
  
  inline uint8_t FEDDAQHeader::version() const
    {
      return ( (header_[0] & 0xF0) >> 4 );
    }
  
  inline bool FEDDAQHeader::hBit() const
    {
      return (header_[0] & 0x8);
    }
  
  inline bool FEDDAQHeader::lastHeader() const
    {
      return !hBit();
    }
  
  inline const uint8_t* FEDDAQHeader::data() const
    {
      return header_;
    }
  
  inline void FEDDAQHeader::print(std::ostream& os) const
    {
      printHex(header_,8,os);
    }

  //FEDDAQTrailer

  inline FEDDAQTrailer::FEDDAQTrailer(const uint8_t* trailer)
    {
      memcpy(trailer_,trailer,8);
    }
  
  inline uint8_t FEDDAQTrailer::eoeNibble() const
    {
      return ( (trailer_[7] & 0xF0) >> 4 );
    }
  
  inline uint32_t FEDDAQTrailer::eventLengthIn64BitWords() const
    {
      return ( trailer_[4] | (trailer_[5]<<8) | (trailer_[6]<<16) );
    }
  
  inline uint32_t FEDDAQTrailer::eventLengthInBytes() const
    {
      return eventLengthIn64BitWords()*8;
    }
  
  inline uint16_t FEDDAQTrailer::crc() const
    {
      return ( trailer_[2] | (trailer_[3]<<8) );
    }
  
  inline bool FEDDAQTrailer::cBit() const
    {
      return (trailer_[1] & 0x80);
    }
  
  inline bool FEDDAQTrailer::fBit() const
    {
      return (trailer_[1] & 0x40);
    }
  
  inline uint8_t FEDDAQTrailer::eventStatusNibble() const
    {
      return (trailer_[1] & 0x0F);
    }
  
  inline uint8_t FEDDAQTrailer::ttsNibble() const
    {
      return ( (trailer_[0] & 0xF0) >> 4);
    }
  
  inline bool FEDDAQTrailer::tBit() const
    {
      return (trailer_[0] & 0x08);
    }
  
  inline bool FEDDAQTrailer::rBit() const
    {
      return (trailer_[0] & 0x04);
    }
  
  inline void FEDDAQTrailer::print(std::ostream& os) const
    {
      printHex(trailer_,8,os);
    }
  
  inline const uint8_t* FEDDAQTrailer::data() const
    {
      return trailer_;
    }

  //FEDBufferBase

  inline void FEDBufferBase::dump(std::ostream& os) const
    {
      printHex(orderedBuffer_,bufferSize_,os);
    }
  
  inline void FEDBufferBase::dumpOriginalBuffer(std::ostream& os) const
    {
      printHex(originalBuffer_,bufferSize_,os);
    }
  
  inline uint16_t FEDBufferBase::calcCRC() const
    {
      return calculateFEDBufferCRC(orderedBuffer_,bufferSize_);
    }
  
  inline FEDDAQHeader FEDBufferBase::daqHeader() const
    {
      return daqHeader_;
    }
  
  inline FEDDAQTrailer FEDBufferBase::daqTrailer() const
    {
      return daqTrailer_;
    }
  
  inline size_t FEDBufferBase::bufferSize() const
    { 
      return bufferSize_;
    }
  
  inline TrackerSpecialHeader FEDBufferBase::trackerSpecialHeader() const
    { 
      return specialHeader_;
    }
  
  inline FEDDAQEventType FEDBufferBase::daqEventType() const
    {
      return daqHeader_.eventType();
    }
  
  inline uint32_t FEDBufferBase::daqLvl1ID() const
    {
      return daqHeader_.l1ID();
    }
  
  inline uint16_t FEDBufferBase::daqBXID() const
    {
      return daqHeader_.bxID();
    }
  
  inline uint16_t FEDBufferBase::daqSourceID() const
    {
      return daqHeader_.sourceID();
    }
  
  inline uint32_t FEDBufferBase::daqEventLengthIn64bitWords() const
    {
      return daqTrailer_.eventLengthIn64BitWords();
    }
  
  inline uint32_t FEDBufferBase::daqEventLengthInBytes() const
    {
      return daqTrailer_.eventLengthInBytes();
    }
  
  inline uint16_t FEDBufferBase::daqCRC() const
    {
      return daqTrailer_.crc();
    }
  
  inline FEDTTSBits FEDBufferBase::daqTTSState() const
    {
      return daqTrailer_.ttsBits();
    }
  
  inline FEDBufferFormat FEDBufferBase::bufferFormat() const
    {
      return specialHeader_.bufferFormat();
    }
  
  inline FEDHeaderType FEDBufferBase::headerType() const
    {
      return specialHeader_.headerType();
    }
  
  inline FEDReadoutMode FEDBufferBase::readoutMode() const
    {
      return specialHeader_.readoutMode();
    }
  
  inline FEDDataType FEDBufferBase::dataType() const
    {
      return specialHeader_.dataType();
    }
  
  inline uint8_t FEDBufferBase::apveAddress() const
    {
      return specialHeader_.apveAddress();
    }
  
  inline bool FEDBufferBase::majorityAddressErrorForFEUnit(const uint8_t internalFEUnitNum) const
    {
      return (specialHeader_.majorityAddressErrorForFEUnit(internalFEUnitNum) && (specialHeader_.apveAddress() != 0x00));
    }
  
  inline bool FEDBufferBase::feEnabled(const uint8_t internalFEUnitNum) const
    {
      return specialHeader_.feEnabled(internalFEUnitNum);
    }
  
  inline bool FEDBufferBase::feOverflow(const uint8_t internalFEUnitNum) const
    {
      return specialHeader_.feOverflow(internalFEUnitNum);
    }
  
  inline FEDStatusRegister FEDBufferBase::fedStatusRegister() const
    {
      return specialHeader_.fedStatusRegister();
    }
  
  inline bool FEDBufferBase::channelGood(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const
    {
      return channelGood(internalFEDChannelNum(internalFEUnitNum,internalChannelNum));
    }
  
  inline const FEDChannel& FEDBufferBase::channel(const uint8_t internalFEDChannelNum) const
    {
      return channels_[internalFEDChannelNum];
    }
  
  inline const FEDChannel& FEDBufferBase::channel(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const
    {
      return channel(internalFEDChannelNum(internalFEUnitNum,internalChannelNum));
    }
  
  inline bool FEDBufferBase::doTrackerSpecialHeaderChecks() const
    {
      return ( checkBufferFormat() &&
               checkHeaderType() &&
               checkReadoutMode() &&
               //checkAPVEAddressValid() &&
               checkNoFEOverflows() ); 
    }
  
  inline bool FEDBufferBase::doDAQHeaderAndTrailerChecks() const
    {
      return ( checkNoSLinkTransmissionError() &&
               checkSourceIDs() &&
               checkNoUnexpectedSourceID() &&
               checkNoExtraHeadersOrTrailers() &&
               checkLengthFromTrailer() );
    }
  
  inline bool FEDBufferBase::checkCRC() const
    {
      return ( checkNoSlinkCRCError() && (calcCRC()==daqCRC()) );
    }
  
  inline bool FEDBufferBase::checkBufferFormat() const
    {
      return (bufferFormat() != BUFFER_FORMAT_INVALID);
    }
  
  inline bool FEDBufferBase::checkHeaderType() const
    {
      return (headerType() != HEADER_TYPE_INVALID);
    }
  
  inline bool FEDBufferBase::checkReadoutMode() const
    {
      return (readoutMode() != READOUT_MODE_INVALID);
    }
  
  inline bool FEDBufferBase::checkAPVEAddressValid() const
    {
      return (apveAddress() <= APV_MAX_ADDRESS);
    }
  
  inline bool FEDBufferBase::checkNoFEOverflows() const
    {
      return !specialHeader_.feOverflowRegister();
    }
  
  inline bool FEDBufferBase::checkNoSlinkCRCError() const
    {
      return !daqTrailer_.slinkCRCError();
    }
  
  inline bool FEDBufferBase::checkNoSLinkTransmissionError() const
    {
      return !daqTrailer_.slinkTransmissionError();
    }
  
  inline bool FEDBufferBase::checkNoUnexpectedSourceID() const
    {
      return !daqTrailer_.badSourceID();
    }
  
  inline bool FEDBufferBase::checkNoExtraHeadersOrTrailers() const
    {
      return ( (daqHeader_.boeNibble() == 0x5) && (daqTrailer_.eoeNibble() == 0xA) );
    }
  
  inline bool FEDBufferBase::checkLengthFromTrailer() const
    {
      return (bufferSize() == daqEventLengthInBytes());
    }
  
  inline const uint8_t* FEDBufferBase::getPointerToDataAfterTrackerSpecialHeader() const
    {
      return orderedBuffer_+16;
    }
  
  inline const uint8_t* FEDBufferBase::getPointerToByteAfterEndOfPayload() const
    {
      return orderedBuffer_+bufferSize_-8;
    }

  //FEDChannel

  inline FEDChannel::FEDChannel(const uint8_t*const data, const size_t offset)
    : data_(data),
      offset_(offset)
    {
      length_ = ( data_[(offset_)^7] + (data_[(offset_+1)^7] << 8) );
    }
  
  inline FEDChannel::FEDChannel(const uint8_t*const data, const size_t offset, const uint16_t length)
    : data_(data),
      offset_(offset),
      length_(length)
    {
    }
  
  inline uint16_t FEDChannel::length() const
    {
      return length_;
    }
  
  inline uint8_t FEDChannel::packetCode() const
    {
      return data_[(offset_+2)^7];
    }
  
  inline const uint8_t* FEDChannel::data() const
    {
      return data_;
    }
  
  inline size_t FEDChannel::offset() const
    {
      return offset_;
    }

}

#endif //ndef EventFilter_SiStripRawToDigi_FEDBufferComponents_H
