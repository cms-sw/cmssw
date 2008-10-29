#ifndef EventFilter_SiStripRawToDigi_FEDBuffer_H
#define EventFilter_SiStripRawToDigi_FEDBuffer_H

#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

//
// Constants
//

namespace sistrip {

  static const uint8_t INVALID=0xFF;

  static const uint8_t APVS_PER_CHANNEL=2;
  static const uint8_t FEUNITS_PER_FED=8;
  static const uint8_t CHANNELS_PER_FEUNIT=12;
  static const uint8_t CHANNELS_PER_FED=FEUNITS_PER_FED*CHANNELS_PER_FEUNIT;
  static const uint8_t APVS_PER_FED=CHANNELS_PER_FED*APVS_PER_CHANNEL;
  static const uint8_t APV_MAX_ADDRESS=192;

  enum BufferFormat { BUFFER_FORMAT_INVALID=INVALID,
		      BUFFER_FORMAT_OLD_VME,
		      BUFFER_FORMAT_OLD_SLINK,
		      BUFFER_FORMAT_NEW };
  //these are the values which appear in the buffer.
  static const uint8_t BUFFER_FORMAT_CODE_OLD = 0xED;
  static const uint8_t BUFFER_FORMAT_CODE_NEW = 0xC5;

  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDHeaderType { HEADER_TYPE_INVALID=INVALID,
		       HEADER_TYPE_FULL_DEBUG=1,
		       HEADER_TYPE_APV_ERROR=2 };

  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDReadoutMode { READOUT_MODE_INVALID=INVALID,
			READOUT_MODE_SCOPE=0x1,
			READOUT_MODE_VIRGIN_RAW=0x2,
			READOUT_MODE_PROC_RAW=0x6,
			READOUT_MODE_ZERO_SUPPRESSED=0xA,
			READOUT_MODE_ZERO_SUPPRESSED_LITE=0xC };

  static const uint8_t PACKET_CODE_SCOPE = 0xE5;
  static const uint8_t PACKET_CODE_VIRGIN_RAW = 0xE6;
  static const uint8_t PACKET_CODE_PROC_RAW = 0xF2;
  static const uint8_t PACKET_CODE_ZERO_SUPPRESSED = 0xEA;

  //enum values are values which appear in buffer. DO NOT CHANGE!
  enum FEDDataType { DATA_TYPE_REAL=0,
		     DATA_TYPE_FAKE=1 };

  //enum values are values which appear in buffer. DO NOT CHANGE!
  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  enum FEDDAQEventType { DAQ_EVENT_TYPE_PHYSICS=0x1,
			 DAQ_EVENT_TYPE_CALIBRATION=0x2,
			 DAQ_EVENT_TYPE_TEST=0x3,
			 DAQ_EVENT_TYPE_TECHNICAL=0x4,
			 DAQ_EVENT_TYPE_SIMULATED=0x5,
			 DAQ_EVENT_TYPE_TRACED=0x6,
			 DAQ_EVENT_TYPE_ERROR=0xF,
			 DAQ_EVENT_TYPE_INVALID=INVALID };

  //enum values are values which appear in buffer. DO NOT CHANGE!
  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  enum FEDTTSBits { TTS_DISCONNECTED1=0x0,
		    TTS_WARN_OVERFLOW=0x1,
		    TTS_OUT_OF_SYNC=0x2,
		    TTS_BUSY=0x4,
		    TTS_READY=0x8,
		    TTS_ERROR=0x12,
		    TTS_DISCONNECTED2=0xF,
		    TTS_INVALID=INVALID };

  //
  // Global function declarations
  //

  //used by these classes
  uint8_t internalFEDChannelNum(uint8_t internalFEUnitNum, uint8_t internalChannelNum);
  void printHex(const void* pointer, size_t length, std::ostream& os);
  //to make enums printable
  std::ostream& operator<<(std::ostream& os, const BufferFormat& value);
  std::ostream& operator<<(std::ostream& os, const FEDHeaderType& value);
  std::ostream& operator<<(std::ostream& os, const FEDReadoutMode& value);
  std::ostream& operator<<(std::ostream& os, const FEDDataType& value);
  std::ostream& operator<<(std::ostream& os, const FEDDAQEventType& value);
  std::ostream& operator<<(std::ostream& os, const FEDTTSBits& value);

  //
  // Class definitions
  //

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQHeader
    {
    public:
      inline FEDDAQHeader() { }
      inline FEDDAQHeader(const uint8_t* header);
      //0x5 in first fragment
      inline uint8_t boeNibble() const { return ( (header_[7] & 0xF0) >> 4 ); }
      inline uint8_t eventTypeNibble() const { return (header_[7] & 0x0F); }
      FEDDAQEventType eventType() const;
      inline uint32_t l1ID() const;
      inline uint16_t bxID() const;
      inline uint16_t sourceID() const;
      inline uint8_t version() const { return ( (header_[0] & 0xF0) >> 4 ); }
      //0 if current header word is last, 1 otherwise
      inline bool hBit() const { return (header_[0] & 0x8); }
      inline bool lastHeader() const { return !hBit(); }
      inline void print(std::ostream& os) const { printHex(header_,8,os); }
    private:
      uint8_t header_[8];
    };

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQTrailer
    {
    public:
      inline FEDDAQTrailer() { }
      inline FEDDAQTrailer(const uint8_t* trailer);
      //0xA in first fragment
      inline uint8_t eoeNibble() const { return ( (trailer_[7] & 0xF0) >> 4 ); }
      inline uint32_t eventLengthIn64BitWords() const;
      inline uint32_t eventLengthInBytes() const { return eventLengthIn64BitWords()*8; }
      inline uint16_t crc() const;
      //set to 1 if FRL detects a transmission error over S-link
      inline bool cBit() const { return (trailer_[1] & 0x80); }
      inline bool slinkTransmissionError() const { return cBit(); }
      //set to 1 if the FED ID is not the one expected by the FRL
      inline bool fBit() const { return (trailer_[1] & 0x40); }
      inline bool badFEDID() const { return fBit(); }
      inline uint8_t eventStatusNibble() const { return (trailer_[1] & 0x0F); }
      inline uint8_t ttsNibble() const { return ( (trailer_[0] & 0xF0) >> 4); }
      FEDTTSBits ttsBits() const;
      //0 if the current trailer is the last, 1 otherwise
      inline bool tBit() const { return (trailer_[0] & 0x08); }
      inline bool lastTrailer() const { return !tBit(); }
      //set to 1 if the S-link sender card detects a CRC error (the CRC it computes is put in the CRC field)
      inline bool rBit() const { return (trailer_[0] & 0x04); }
      inline bool slinkCRCError() const { return rBit(); }
      inline void print(std::ostream& os) const { printHex(trailer_,8,os); }
    private:
      uint8_t trailer_[8];
    };

  class FEDStatusRegister
    {
    public:
      inline FEDStatusRegister(const uint16_t fedStatusRegister)
	: data_(fedStatusRegister) { }
      inline bool slinkFullFlag() const { return getBit(0); }
      inline bool trackerHeaderMonitorDataReadyFlag() const { return getBit(1); }
      inline bool qdrMemoryFullFlag() const { return getBit(2); }
      inline bool qdrMemoryPartialFullFlag() const { return getBit(3); }
      inline bool qdrMemoryEmptyFlag() const { return getBit(4); }
      inline bool l1aBxFIFOFullFlag() const { return getBit(5); }
      inline bool l1aBxFIFOPartialFullFlag() const { return getBit(6); }
      inline bool l1aBxFIFOEmptyFlag() const { return getBit(7); }
      inline void print(std::ostream& os) const { printHex(&data_,2,os); }
      void printFlags(std::ostream& os) const;
    private:
      inline bool getBit(const uint8_t num) const { return ( (0x1<<num) & (data_) ); }
      const uint16_t data_;
    };

  class TrackerSpecialHeader
    {
    public:
      inline TrackerSpecialHeader();
      //construct with a pointer to the data. The data will be coppied and swapped if necessary. 
      TrackerSpecialHeader(const uint8_t* headerPointer);
      inline const uint8_t* getPointer() const { return specialHeader_; }
      inline uint8_t bufferFormatByte() const { return specialHeader_[BUFFERFORMAT]; }
      BufferFormat bufferFormat() const;
      inline uint8_t headerTypeNibble() const { return ( (specialHeader_[BUFFERTYPE] & 0xF0) >> 4 ); }
      FEDHeaderType headerType() const;
      inline uint8_t trackerEventTypeNibble() const { return (specialHeader_[BUFFERTYPE] & 0x0F); }
      FEDReadoutMode readoutMode() const;
      FEDDataType dataType() const;
      inline uint8_t apveAddress() const { return specialHeader_[APVEADDRESS]; }
      inline uint8_t apvAddressErrorRegister() const { return specialHeader_[ADDRESSERROR]; }
      inline bool majorityAddressErrorForFEUnit(uint8_t internalFEUnitNum) const;
      inline uint8_t feEnableRegister() const { return specialHeader_[FEENABLE]; }
      inline bool feEnabled(uint8_t internalFEUnitNum) const;
      inline uint8_t feOverflowRegister() const { return specialHeader_[FEOVERFLOW]; }
      inline bool feOverflow(uint8_t internalFEUnitNum) const;
      inline uint16_t fedStatusRegisterWord() const;
      inline FEDStatusRegister fedStatusRegister() const { return FEDStatusRegister(fedStatusRegisterWord()); }
      inline void print(std::ostream& os) const { printHex(specialHeader_,8,os); }
    private:
      enum byteIndicies { FEDSTATUS=0, FEOVERFLOW=2, FEENABLE=3, ADDRESSERROR=4, APVEADDRESS=5, BUFFERTYPE=6, BUFFERFORMAT=7 };
      //copy of header, 32 bit word swapped if needed
      uint8_t specialHeader_[8];
      //was the header word swapped?
      bool wordSwapped_;
    };

  class FEDBufferBase
    {
    public:
      FEDBufferBase(const uint8_t* fedBuffer, size_t fedBufferSize, bool allowUnrecognizedFormat = false);
      virtual ~FEDBufferBase();
      //dump buffer to stream
      inline void dump(std::ostream& os) const { printHex(orderedBuffer_,bufferSize_,os); }
      //dump original buffer before word swapping
      inline void dumpOriginalBuffer(std::ostream& os) const { printHex(originalBuffer_,bufferSize_,os); }
      void print(std::ostream& os) const;
      //calculate the CRC from the buffer
      uint16_t calcCRC() const;
  
      //methods to get parts of the buffer
      inline FEDDAQHeader daqHeader() const { return daqHeader_; }
      inline FEDDAQTrailer daqTrailer() const { return daqTrailer_; }
      inline size_t bufferSize() const { return bufferSize_; }
      inline TrackerSpecialHeader trackerSpecialHeader() const { return specialHeader_; }
      //methods to get info from DAQ header
      inline FEDDAQEventType daqEventType() const { return daqHeader_.eventType(); }
      inline uint32_t daqLvl1ID() const { return daqHeader_.l1ID(); }
      inline uint16_t daqBXID() const { return daqHeader_.bxID(); }
      inline uint16_t daqSourceID() const { return daqHeader_.sourceID(); }
      inline uint16_t sourceID() const { return daqSourceID(); }
      //methods to get info from DAQ trailer
      inline uint32_t daqEventLengthIn64bitWords() const { return daqTrailer_.eventLengthIn64BitWords(); }
      inline uint32_t daqEventLengthInBytes() const { return daqTrailer_.eventLengthInBytes(); }
      inline uint16_t daqCRC() const { return daqTrailer_.crc(); }
      inline FEDTTSBits daqTTSState() const { return daqTrailer_.ttsBits(); }
      //methods to get info from the tracker special header
      inline BufferFormat bufferFormat() const { return specialHeader_.bufferFormat(); }
      inline FEDHeaderType headerType() const { return specialHeader_.headerType(); }
      inline FEDReadoutMode readoutMode() const { return specialHeader_.readoutMode(); }
      inline FEDDataType dataType() const { return specialHeader_.dataType(); }
      inline uint8_t apveAddress() const { return specialHeader_.apveAddress(); }
      inline bool majorityAddressErrorForFEUnit(uint8_t internalFEUnitNum) const { return specialHeader_.majorityAddressErrorForFEUnit(internalFEUnitNum); }
      inline bool feEnabled(uint8_t internalFEUnitNum) const { return specialHeader_.feEnabled(internalFEUnitNum); }
      uint8_t nFEUnitsEnabled() const;
      inline bool feOverflow(uint8_t internalFEUnitNum) const { return specialHeader_.feOverflow(internalFEUnitNum); }
      inline bool feGood(uint8_t internalFEUnitNum) const { return ( !majorityAddressErrorForFEUnit(internalFEUnitNum) && !feOverflow(internalFEUnitNum) ); }
      inline FEDStatusRegister fedStatusRegister() const { return specialHeader_.fedStatusRegister(); }
  
      //summary checks
      //check that tracker special header is valid (does not check for FE unit errors indicated in special header)
      inline bool doTrackerSpecialHeaderChecks() const;
      //check for errors in DAQ heaqder and trailer (not including bad CRC)
      inline bool doDAQHeaderAndTrailerChecks() const;
      //do both
      virtual bool doChecks() const;
      //print the result of all detailed checks
      virtual std::string checkSummary() const;
  
      //detailed checks
      inline bool checkCRC() const { return ( checkNoSlinkCRCError() && (calcCRC()==daqCRC()) ); }
      //methods to check tracker special header
      inline bool checkBufferFormat() const { return (bufferFormat() != BUFFER_FORMAT_INVALID); }
      inline bool checkHeaderType() const { return (headerType() != HEADER_TYPE_INVALID); }
      inline bool checkReadoutMode() const { return (readoutMode() != READOUT_MODE_INVALID); }
      inline bool checkAPVEAddressValid() const { return (apveAddress() <= APV_MAX_ADDRESS); }
      bool checkMajorityAddresses() const;
      inline bool checkNoFEOverflows() const { return !specialHeader_.feOverflowRegister(); }
      //methods to check daq header and trailer
      inline bool checkNoSlinkCRCError() const { return !daqTrailer_.slinkCRCError(); }
      inline bool checkNoSLinkTransmissionError() const { return !daqTrailer_.slinkTransmissionError(); }
      bool checkSourceIDs() const;
      inline bool checkNoUnexpectedSourceID() const { return !daqTrailer_.badFEDID(); }
      inline bool checkNoExtraHeadersOrTrailers() const { return ( (daqHeader_.boeNibble() == 0x5) && (daqTrailer_.eoeNibble() == 0xA) ); }
      inline bool checkLengthFromTrailer() const { return (bufferSize() == daqEventLengthInBytes()); }
    protected:
      inline const uint8_t* getPointerToDataAfterTrackerSpecialHeader() const
	{ return orderedBuffer_+16; }
      inline uint8_t* getPointerToDataAfterTrackerSpecialHeader();
      inline const uint8_t* getPointerToByteAfterEndOfPayload() const
	{ return orderedBuffer_+bufferSize_-8; }
      inline uint8_t* getPointerToByteAfterEndOfPayload();
    private:
      const uint8_t* originalBuffer_;
      const uint8_t* orderedBuffer_;
      const size_t bufferSize_;
      FEDDAQHeader daqHeader_;
      FEDDAQTrailer daqTrailer_;
      TrackerSpecialHeader specialHeader_;
    };

  class FEDBackendStatusRegister
    {
    public:
      inline FEDBackendStatusRegister(const uint32_t backendStatusRegister)
	: data_(backendStatusRegister) { }
      inline bool internalFreezeFlag() const { return getBit(1); }
      inline bool slinkDownFlag() const { return getBit(2); }
      inline bool slinkFullFlag() const { return getBit(3); }
      inline bool backpressureFlag() const { return getBit(4); }
      //bit 5 undefined
      inline bool ttcReadyFlag() const { return getBit(6); }
      inline bool trackerHeaderMonitorDataReadyFlag() const { return getBit(7); }
      inline bool qdrMemoryFullFlag() const { return getBit(8); }
      inline bool frameAddressFIFOFullFlag() const { return getBit(9); }
      inline bool totalLengthFIFOFullFlag() const { return getBit(10); }
      inline bool trackerHeaderFIFOFullFlag() const { return getBit(11); }
      inline bool l1aBxFIFOFullFlag() const { return getBit(12); }
      inline bool feEventLengthFIFOFullFlag() const { return getBit(13); }
      inline bool feFPGAFullFlag() const { return getBit(14); }
      //bit 15 undefined
      inline bool qdrMemoryPartialFullFlag() const { return getBit(16); }
      inline bool frameAddressFIFOPartialFullFlag() const { return getBit(17); }
      inline bool totalLengthFIFOPartialFullFlag() const { return getBit(18); }
      inline bool trackerHeaderFIFOPartialFullFlag() const { return getBit(19); }
      inline bool l1aBxFIFOPartialFullFlag() const { return getBit(20); }
      inline bool feEventLengthFIFOPartialFullFlag() const { return getBit(21); }
      inline bool feFPGAPartialFullFlag() const { return getBit(22); }
      //bit 22 undefined
      inline bool qdrMemoryEmptyFlag() const { return getBit(24); }
      inline bool frameAddressFIFOEmptyFlag() const { return getBit(25); }
      inline bool totalLengthFIFOEmptyFlag() const { return getBit(26); }
      inline bool trackerHeaderFIFOEmptyFlag() const { return getBit(27); }
      inline bool l1aBxFIFOEmptyFlag() const { return getBit(28); }
      inline bool feEventLengthFIFOEmptyFlag() const { return getBit(29); }
      inline bool feFPGAEmptyFlag() const { return getBit(30); }
      //bit 31 undefined
      inline void print(std::ostream& os) const { printHex(&data_,4,os); }
      void printFlags(std::ostream& os) const;
    private:
      inline bool getBit(const uint8_t num) const { return ( (0x1<<num) & (data_) ); }
      const uint32_t data_;
    };

  class FEDFEHeader
    {
    public:
      //factory function: allocates new FEDFEHeader derrivative of appropriate type
      inline static FEDFEHeader* newFEHeader(FEDHeaderType headerType, const uint8_t* headerBuffer);
      virtual ~FEDFEHeader();
      //the length of the header
      virtual size_t lengthInBytes() const = 0;
      //check that there are no errors indicated in which ever error bits are available in the header
      //check bits for both APVs on a channel
      inline bool checkChannelStatusBits(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return checkChannelStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      virtual bool checkChannelStatusBits(uint8_t internalFEDChannelNum) const = 0;
      //check bits for one APV
      inline bool checkStatusBits(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return checkStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
      virtual bool checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const = 0;
      virtual void print(std::ostream& os) const = 0;
    };

  class FEDAPVErrorHeader : public FEDFEHeader
    {
    public:
      inline FEDAPVErrorHeader(const uint8_t* headerBuffer);
      virtual ~FEDAPVErrorHeader();
      virtual size_t lengthInBytes() const;
      virtual bool checkChannelStatusBits(uint8_t internalFEDChannelNum) const;
      virtual bool checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const;
      virtual void print(std::ostream& os) const;
    private:
      static const size_t APV_ERROR_HEADER_SIZE_IN_64BIT_WORDS = 3;
      static const size_t APV_ERROR_HEADER_SIZE_IN_BYTES = APV_ERROR_HEADER_SIZE_IN_64BIT_WORDS*8;
      uint8_t header_[APV_ERROR_HEADER_SIZE_IN_BYTES];
    };

  class FEDFullDebugHeader : public FEDFEHeader
    {
    public:
      inline FEDFullDebugHeader(const uint8_t* headerBuffer);
      virtual ~FEDFullDebugHeader();
      virtual size_t lengthInBytes() const;
      virtual bool checkChannelStatusBits(uint8_t internalFEDChannelNum) const;
      virtual bool checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const;
      virtual void print(std::ostream& os) const;
  
      inline uint8_t feUnitMajorityAddress(uint8_t internalFEUnitNum) const { return feWord(internalFEUnitNum)[9]; }
      inline FEDBackendStatusRegister beStatusRegister() const { return FEDBackendStatusRegister(get32BitWordFrom(feWord(0)+10)); }
      inline uint32_t daqRegister() const { return get32BitWordFrom(feWord(7)+10); }
      inline uint32_t daqRegister2() const { return get32BitWordFrom(feWord(6)+10); }
      inline uint16_t feUnitLength(uint8_t internalFEUnitNum) const { return ( (feWord(internalFEUnitNum)[15]<<8) | (feWord(internalFEUnitNum)[14]) ); }
  
      //These methods return true if there was an error of the appropriate type (ie if the error bit is 0).
      //They return false if the error could not occur due to a more general error.
      //was channel unlocked
      inline bool unlocked(uint8_t internalFEDChannelNum) const { return unlockedFromBit(internalFEDChannelNum); }
      inline bool unlocked(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return unlocked(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //was channel out of sync if it was unlocked
      inline bool outOfSync(uint8_t internalFEDChannelNum) const
	{ return ( !unlocked(internalFEDChannelNum) && outOfSyncFromBit(internalFEDChannelNum) ); }
      inline bool outOfSync(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return outOfSync(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //was there an internal APV error if it was in sync
      inline bool apvError(uint8_t internalFEDChannelNum, uint8_t apvNum) const
	{ return ( !outOfSync(internalFEDChannelNum) && apvErrorFromBit(internalFEDChannelNum,apvNum) ); }
      inline bool apvError(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvError(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
      //was the APV address wrong if it was in sync (does not depend on APV internal error bit)
      inline bool apvAddressError(uint8_t internalFEDChannelNum, uint8_t apvNum) const
	{ return ( !outOfSync(internalFEDChannelNum) && apvAddressErrorFromBit(internalFEDChannelNum,apvNum) ); }
      inline bool apvAddressError(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvAddressError(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
  
      //These methods return true if there was an error of the appropriate type (ie if the error bit is 0).
      //They ignore any previous errors which make the status bits meaningless and return the value of the bit anyway.
      //In general, the methods above which only return an error for the likely cause are more useful.
      inline bool unlockedFromBit(uint8_t internalFEDChannelNum) const { return !getBitVal(internalFEDChannelNum,5); }
      inline bool unlockedFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return unlockedFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      inline bool outOfSyncFromBit(uint8_t internalFEDChannelNum) const { return !getBitVal(internalFEDChannelNum,4); }
      inline bool outOfSyncFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return outOfSyncFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      inline bool apvErrorFromBit(uint8_t internalFEDChannelNum, uint8_t apvNum) const { return !getBitVal(internalFEDChannelNum,0+2*apvNum); }
      inline bool apvErrorFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvErrorFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
      inline bool apvAddressErrorFromBit(uint8_t internalFEDChannelNum, uint8_t apvNum) const { return !getBitVal(internalFEDChannelNum,1+2*apvNum); }
      inline bool apvAddressErrorFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvAddressErrorFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
    private:
      inline bool getBitVal(uint8_t internalFEDChannelNum, uint8_t bit) const;
      static inline uint32_t get32BitWordFrom(const uint8_t* startOfWord);
      inline const uint8_t* feWord(uint8_t internalFEUnitNum) const { return header_+internalFEUnitNum*2*8; }
      static const size_t FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS = FEUNITS_PER_FED*2;
      static const size_t FULL_DEBUG_HEADER_SIZE_IN_BYTES = FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS*8;
      uint8_t header_[FULL_DEBUG_HEADER_SIZE_IN_BYTES];
    };

  class FEDZSChannelUnpacker;
  class FEDRawChannelUnpacker;

  class FEDChannel
    {
    public:
      inline FEDChannel(const uint8_t* data, size_t offset);
      inline uint16_t length() const { return length_; }
      inline uint8_t packetCode() const { return data_[(offset_+2)^7]; }
    private:
      friend class FEDBuffer;
      friend class FEDZSChannelUnpacker;
      friend class FEDRawChannelUnpacker;
      inline const uint8_t* data() const { return data_; }
      inline size_t offset() const { return offset_; }
      const uint8_t* data_;
      uint16_t length_;
      size_t offset_;
    };

  class FEDBuffer : public FEDBufferBase
    {
    public:
      //construct from buffer
      //if allowBadBuffer is set to true then exceptions will not be thrown if the channel lengths do not make sense or the event format is not recognized
      FEDBuffer(const uint8_t* fedBuffer, size_t fedBufferSize, bool allowBadBuffer = false);
      virtual ~FEDBuffer();
      inline const FEDFEHeader* feHeader() const { return feHeader_; }
      //check that channel is on enabled FE Unit and has no errors
      inline bool channelGood(uint8_t internalFEDChannelNum) const;
      inline bool channelGood(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return channelGood(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //return channel object for channel
      inline const FEDChannel& channel(uint8_t internalFEDChannelNum) const { return channels_[internalFEDChannelNum]; }
      inline const FEDChannel& channel(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return channel(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }

      //functions to check buffer. All return true if there is no problem.
      //minimum checks to do before using buffer
      virtual bool doChecks() const;
  
      //additional checks to check for corrupt buffers
      //check channel lengths fit inside to buffer length
      bool checkChannelLengths() const;
      //check that channel lengths add up to buffer length (this does the previous check as well)
      bool checkChannelLengthsMatchBufferLength() const;
      //check channel packet codes match readout mode
      bool checkChannelPacketCodes() const;
      //check FE unit lengths in FULL DEBUG header match the lengths of their channels
      bool checkFEUnitLengths() const;
      //check FE unit APV addresses in FULL DEBUG header are equal to the APVe address if the majority was good
      bool checkFEUnitAPVAddresses() const;
      //do all corrupt buffer checks
      virtual bool doCorruptBufferChecks() const;
  
      //check that there are no errors in channel, APV or FEUnit status bits
      //these are done by channelGood(). Channels with bad status bits may be disabled so bad status bits do not usually indicate an error
      inline bool checkStatusBits(uint8_t internalFEDChannelNum) const { return feHeader_->checkChannelStatusBits(internalFEDChannelNum); }
      inline bool checkStatusBits(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return checkStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //same but for all channels on enabled FE units
      bool checkAllChannelStatusBits() const;
  
      //print a summary of all checks
      virtual std::string checkSummary() const;
    private:
      void findChannels();
      std::vector<FEDChannel> channels_;
      const FEDFEHeader* feHeader_;
      uint8_t* payloadPointer_;
      uint16_t payloadLength_;
      uint8_t lastValidChannel_;
    };

  class FEDZSChannelUnpacker
    {
    public:
      static inline FEDZSChannelUnpacker zeroSuppressedModeUnpacker(const FEDChannel& channel);
      static inline FEDZSChannelUnpacker zeroSuppressedLiteModeUnpacker(const FEDChannel& channel);
      inline uint8_t strip() const { return currentStrip_; }
      inline uint8_t adc() const { return data_[currentOffset_^7]; }
      inline bool hasData() const { return (valuesLeftAfterThisCluster_ || valuesLeftInCluster_); }
      inline FEDZSChannelUnpacker& operator ++ ();
    private:
      inline FEDZSChannelUnpacker(const uint8_t* data, size_t currentOffset,
				  uint8_t currentStrip, uint8_t valuesLeftInCluster,
				  int16_t valuesLeftAfterThisCluster);
      inline void readNewClusterInfo();
      static void throwBadChannelLength(uint16_t length);
      void throwBadClusterLength();
      const uint8_t* data_;
      size_t currentOffset_;
      uint8_t currentStrip_;
      uint8_t valuesLeftInCluster_;
      int16_t valuesLeftAfterThisCluster_;
    };

  class FEDRawChannelUnpacker
    {
    public:
      static inline FEDRawChannelUnpacker scopeModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      static inline FEDRawChannelUnpacker virginRawModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      static inline FEDRawChannelUnpacker procRawModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      inline FEDRawChannelUnpacker(const FEDChannel& channel);
      inline uint8_t strip() const { return currentStrip_; }
      inline uint16_t adc() const { return ( data_[currentOffset_^7] + ((data_[(currentOffset_+1)^7]&0x03)<<8) ); }
      inline bool hasData() const { return valuesLeft_; }
      inline FEDRawChannelUnpacker& operator ++ ();
    private:
      static void throwBadChannelLength(uint16_t length);
      const uint8_t* data_;
      size_t currentOffset_;
      uint8_t currentStrip_;
      uint8_t valuesLeft_;
    };

  //
  // Inline function definitions
  //

  inline uint8_t internalFEDChannelNum(uint8_t internalFEUnitNum, uint8_t internalChannelNum)
    {
      return (internalFEUnitNum*CHANNELS_PER_FEUNIT + internalChannelNum);
    }

  inline std::ostream& operator << (std::ostream& os, const FEDDAQHeader& data) { data.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDDAQTrailer& data) { data.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const TrackerSpecialHeader& data) { data.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDStatusRegister& data) { data.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDFEHeader& data) { data.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDBufferBase& data) { data.print(os); return os; }

  //FEDBuffer

  inline bool FEDBuffer::channelGood(uint8_t internalFEDChannelNum) const
    {
      return ( (internalFEDChannelNum <= lastValidChannel_) &&
	       feGood(internalFEDChannelNum/CHANNELS_PER_FEUNIT) &&
	       checkStatusBits(internalFEDChannelNum) );
    }

  //FEDBufferBase

  inline bool FEDBufferBase::doTrackerSpecialHeaderChecks() const
    {
      return ( checkBufferFormat() &&
	       checkHeaderType() &&
	       checkReadoutMode() &&
	       checkAPVEAddressValid() &&
	       checkNoFEOverflows() && 
	       checkMajorityAddresses() );
    }

  inline bool FEDBufferBase::doDAQHeaderAndTrailerChecks() const
    {
      return ( checkNoSLinkTransmissionError() &&
	       checkSourceIDs() &&
	       checkNoUnexpectedSourceID() &&
	       checkNoExtraHeadersOrTrailers() &&
	       checkLengthFromTrailer() );
    }

  //re-use the const method by using static and const casts to avoid code duplication
  inline uint8_t* FEDBufferBase::getPointerToDataAfterTrackerSpecialHeader()
    {
      const FEDBufferBase* constThis = static_cast<const FEDBufferBase*>(this);
      const uint8_t* constPointer = constThis->getPointerToDataAfterTrackerSpecialHeader();
      return const_cast<uint8_t*>(constPointer);
    }

  inline uint8_t* FEDBufferBase::getPointerToByteAfterEndOfPayload()
    {
      const FEDBufferBase* constThis = static_cast<const FEDBufferBase*>(this);
      const uint8_t* constPointer = constThis->getPointerToByteAfterEndOfPayload();
      return const_cast<uint8_t*>(constPointer);
    }

  //TrackerSpecialHeader

  inline TrackerSpecialHeader::TrackerSpecialHeader()
    : wordSwapped_(false)
    {
    }

  inline bool TrackerSpecialHeader::majorityAddressErrorForFEUnit(uint8_t internalFEUnitNum) const
    {
      //TODO: check this is correct order
      return ( (0x1<<internalFEUnitNum) & apvAddressErrorRegister() );
    }

  inline bool TrackerSpecialHeader::feEnabled(uint8_t internalFEUnitNum) const
    {
      //TODO: check this is correct order
      return ( (0x1<<internalFEUnitNum) & feEnableRegister() );
    }

  inline bool TrackerSpecialHeader::feOverflow(uint8_t internalFEUnitNum) const
    {
      //TODO: check this is correct order
      return ( (0x1<<internalFEUnitNum) & feOverflowRegister() );
    }

  inline uint16_t TrackerSpecialHeader::fedStatusRegisterWord() const
    {
      //get 16 bits
      //TODO: Is this the correct byte ordering
      uint16_t statusRegister = ( (specialHeader_[(FEDSTATUS+1)]<<8) | specialHeader_[FEDSTATUS]);
      return statusRegister;
    }

  //FEDFEHeader

  inline FEDFEHeader* FEDFEHeader::newFEHeader(FEDHeaderType headerType, const uint8_t* headerBuffer)
    {
      switch (headerType) {
      case HEADER_TYPE_FULL_DEBUG:
	return new FEDFullDebugHeader(headerBuffer);
      case HEADER_TYPE_APV_ERROR:
	return new FEDAPVErrorHeader(headerBuffer);
      default:
	//TODO: throw exception
	return NULL;
      }
    }

  inline FEDAPVErrorHeader::FEDAPVErrorHeader(const uint8_t* headerBuffer)
    {
      memcpy(header_,headerBuffer,APV_ERROR_HEADER_SIZE_IN_BYTES);
    }

  inline FEDFullDebugHeader::FEDFullDebugHeader(const uint8_t* headerBuffer)
    {
      memcpy(header_,headerBuffer,FULL_DEBUG_HEADER_SIZE_IN_BYTES);
    }

  inline bool FEDFullDebugHeader::getBitVal(uint8_t internalFEDChannelNum, uint8_t bit) const
    {
      const uint8_t* pFEWord = feWord(internalFEDChannelNum / CHANNELS_PER_FEUNIT);
      const uint8_t bitInFeWord = (internalFEDChannelNum % CHANNELS_PER_FEUNIT) * 6 + bit;
      return ( pFEWord[bitInFeWord/8] & (0x1 << bitInFeWord%8) );
    }

  inline uint32_t FEDFullDebugHeader::get32BitWordFrom(const uint8_t* startOfWord)
    {
      //TODO: check byte ordering
      return ( startOfWord[0] | (startOfWord[1]<<8) | (startOfWord[2]<<16) | (startOfWord[3]<<24) );
    }

  //FEDDAQHeader

  inline FEDDAQHeader::FEDDAQHeader(const uint8_t* header)
    {
      memcpy(header_,header,8);
    }

  inline uint32_t FEDDAQHeader::l1ID() const
    {
      //TODO: check byte ordering
      return ( header_[4] | (header_[5]<<8) | (header_[6]<<16) );
    }

  inline uint16_t FEDDAQHeader::bxID() const
    {
      //TODO: check byte ordering
      return ( (header_[3]<<4) | ((header_[2]&0xF0)>>4) );
    }

  inline uint16_t FEDDAQHeader::sourceID() const
    {
      //TODO: check byte ordering
      return ( ((header_[2]&0x0F)<<8) | header_[1] );
    }

  //FEDDAQTrailer

  inline FEDDAQTrailer::FEDDAQTrailer(const uint8_t* trailer)
    {
      memcpy(trailer_,trailer,8);
    }

  inline uint32_t FEDDAQTrailer::eventLengthIn64BitWords() const
    {
      //TODO: check byte ordering
      return ( trailer_[4] | (trailer_[5]<<8) | (trailer_[6]<<16) );
    }

  inline uint16_t FEDDAQTrailer::crc() const
    {
      //TODO: check byte ordering
      return ( trailer_[2] | (trailer_[3]<<8) );
    }

  //FEDChannel

  FEDChannel::FEDChannel(const uint8_t* data, size_t offset)
    : data_(data),
    offset_(offset)
    {
      length_ = ( data_[(offset_)^7] + (data_[(offset_+1)^7] << 8) );
    }

  //FEDRawChannelUnpacker

  inline FEDRawChannelUnpacker::FEDRawChannelUnpacker(const FEDChannel& channel)
    : data_(channel.data()),
    currentOffset_(channel.offset()+3),
    currentStrip_(0),
    valuesLeft_((channel.length()-3)/2)
    {
      if ((channel.length()-3)%2) throwBadChannelLength(channel.length());
    }

  inline FEDRawChannelUnpacker& FEDRawChannelUnpacker::operator ++ ()
    {
      currentOffset_ += 2;
      currentStrip_++;
      valuesLeft_--;
      return (*this);
    }

  //FEDZSChannelUnpacker

  inline FEDZSChannelUnpacker::FEDZSChannelUnpacker(const uint8_t* data, size_t currentOffset,
						    uint8_t currentStrip, uint8_t valuesLeftInCluster,
						    int16_t valuesLeftAfterThisCluster)
    : data_(data),
    currentOffset_(currentOffset),
    currentStrip_(currentStrip),
    valuesLeftInCluster_(valuesLeftInCluster),
    valuesLeftAfterThisCluster_(valuesLeftAfterThisCluster)
    {
    }

  inline void FEDZSChannelUnpacker::readNewClusterInfo()
    {
      if (valuesLeftAfterThisCluster_) {
	currentStrip_ = data_[(currentOffset_++)^7];
	valuesLeftInCluster_ = data_[(currentOffset_++)^7];
	valuesLeftAfterThisCluster_ -= valuesLeftInCluster_+2;
	if (valuesLeftAfterThisCluster_ < 0) throwBadClusterLength();
      }
    }

  inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ ()
    {
      if (valuesLeftInCluster_) {
	currentStrip_++;
	currentOffset_++;
      } else {
	currentOffset_++;
	readNewClusterInfo();
      }
      return (*this);
    }

  inline FEDZSChannelUnpacker FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(const FEDChannel& channel)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDZSChannelUnpacker result(channel.data(),channel.offset()+7,0,0,length-7);
      result.readNewClusterInfo();
      return result;
    }

  inline FEDZSChannelUnpacker FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(const FEDChannel& channel)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDZSChannelUnpacker result(channel.data(),channel.offset()+2,0,0,length-2);
      result.readNewClusterInfo();
      return result;
    }

}

#endif //EventFilter_SiStripRawToDigi_FEDBuffer_H
