#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBufferComponents_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBufferComponents_H

#include "boost/cstdint.hpp"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include <ostream>

namespace sistrip {
  
  //
  // Constants
  //

  static const uint8_t INVALID=0xFF;

  static const uint8_t APV_MAX_ADDRESS=192;

  enum FEDBufferFormat { BUFFER_FORMAT_INVALID=INVALID,
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
  std::ostream& operator<<(std::ostream& os, const FEDBufferFormat& value);
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
      FEDDAQHeader() { }
      FEDDAQHeader(const uint8_t* header);
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
    private:
      uint8_t header_[8];
    };

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQTrailer
    {
    public:
      FEDDAQTrailer() { }
      FEDDAQTrailer(const uint8_t* trailer);
      //0xA in first fragment
      uint8_t eoeNibble() const { return ( (trailer_[7] & 0xF0) >> 4 ); }
      uint32_t eventLengthIn64BitWords() const;
      uint32_t eventLengthInBytes() const { return eventLengthIn64BitWords()*8; }
      uint16_t crc() const;
      //set to 1 if FRL detects a transmission error over S-link
      bool cBit() const { return (trailer_[1] & 0x80); }
      bool slinkTransmissionError() const { return cBit(); }
      //set to 1 if the FED ID is not the one expected by the FRL
      bool fBit() const { return (trailer_[1] & 0x40); }
      bool badFEDID() const { return fBit(); }
      uint8_t eventStatusNibble() const { return (trailer_[1] & 0x0F); }
      uint8_t ttsNibble() const { return ( (trailer_[0] & 0xF0) >> 4); }
      FEDTTSBits ttsBits() const;
      //0 if the current trailer is the last, 1 otherwise
      bool tBit() const { return (trailer_[0] & 0x08); }
      bool lastTrailer() const { return !tBit(); }
      //set to 1 if the S-link sender card detects a CRC error (the CRC it computes is put in the CRC field)
      bool rBit() const { return (trailer_[0] & 0x04); }
      bool slinkCRCError() const { return rBit(); }
      void print(std::ostream& os) const { printHex(trailer_,8,os); }
    private:
      uint8_t trailer_[8];
    };

  class FEDStatusRegister
    {
    public:
      FEDStatusRegister(const uint16_t fedStatusRegister)
	: data_(fedStatusRegister) { }
      bool slinkFullFlag() const { return getBit(0); }
      bool trackerHeaderMonitorDataReadyFlag() const { return getBit(1); }
      bool qdrMemoryFullFlag() const { return getBit(2); }
      bool qdrMemoryPartialFullFlag() const { return getBit(3); }
      bool qdrMemoryEmptyFlag() const { return getBit(4); }
      bool l1aBxFIFOFullFlag() const { return getBit(5); }
      bool l1aBxFIFOPartialFullFlag() const { return getBit(6); }
      bool l1aBxFIFOEmptyFlag() const { return getBit(7); }
      void print(std::ostream& os) const { printHex(&data_,2,os); }
      void printFlags(std::ostream& os) const;
    private:
      bool getBit(const uint8_t num) const { return ( (0x1<<num) & (data_) ); }
      const uint16_t data_;
    };

  class TrackerSpecialHeader
    {
    public:
      TrackerSpecialHeader();
      //construct with a pointer to the data. The data will be coppied and swapped if necessary. 
      TrackerSpecialHeader(const uint8_t* headerPointer);
      const uint8_t* getPointer() const { return specialHeader_; }
      uint8_t bufferFormatByte() const { return specialHeader_[BUFFERFORMAT]; }
      FEDBufferFormat bufferFormat() const;
      uint8_t headerTypeNibble() const { return ( (specialHeader_[BUFFERTYPE] & 0xF0) >> 4 ); }
      FEDHeaderType headerType() const;
      uint8_t trackerEventTypeNibble() const { return (specialHeader_[BUFFERTYPE] & 0x0F); }
      FEDReadoutMode readoutMode() const;
      FEDDataType dataType() const;
      uint8_t apveAddress() const { return specialHeader_[APVEADDRESS]; }
      uint8_t apvAddressErrorRegister() const { return specialHeader_[ADDRESSERROR]; }
      bool majorityAddressErrorForFEUnit(uint8_t internalFEUnitNum) const;
      uint8_t feEnableRegister() const { return specialHeader_[FEENABLE]; }
      bool feEnabled(uint8_t internalFEUnitNum) const;
      uint8_t feOverflowRegister() const { return specialHeader_[FEOVERFLOW]; }
      bool feOverflow(uint8_t internalFEUnitNum) const;
      uint16_t fedStatusRegisterWord() const;
      FEDStatusRegister fedStatusRegister() const { return FEDStatusRegister(fedStatusRegisterWord()); }
      void print(std::ostream& os) const { printHex(specialHeader_,8,os); }
    private:
      enum byteIndicies { FEDSTATUS=0, FEOVERFLOW=2, FEENABLE=3, ADDRESSERROR=4, APVEADDRESS=5, BUFFERTYPE=6, BUFFERFORMAT=7 };
      //copy of header, 32 bit word swapped if needed
      uint8_t specialHeader_[8];
      //was the header word swapped?
      bool wordSwapped_;
    };

  class FEDBackendStatusRegister
    {
    public:
      FEDBackendStatusRegister(const uint32_t backendStatusRegister)
	: data_(backendStatusRegister) { }
      bool internalFreezeFlag() const { return getBit(1); }
      bool slinkDownFlag() const { return getBit(2); }
      bool slinkFullFlag() const { return getBit(3); }
      bool backpressureFlag() const { return getBit(4); }
      //bit 5 undefined
      bool ttcReadyFlag() const { return getBit(6); }
      bool trackerHeaderMonitorDataReadyFlag() const { return getBit(7); }
      bool qdrMemoryFullFlag() const { return getBit(8); }
      bool frameAddressFIFOFullFlag() const { return getBit(9); }
      bool totalLengthFIFOFullFlag() const { return getBit(10); }
      bool trackerHeaderFIFOFullFlag() const { return getBit(11); }
      bool l1aBxFIFOFullFlag() const { return getBit(12); }
      bool feEventLengthFIFOFullFlag() const { return getBit(13); }
      bool feFPGAFullFlag() const { return getBit(14); }
      //bit 15 undefined
      bool qdrMemoryPartialFullFlag() const { return getBit(16); }
      bool frameAddressFIFOPartialFullFlag() const { return getBit(17); }
      bool totalLengthFIFOPartialFullFlag() const { return getBit(18); }
      bool trackerHeaderFIFOPartialFullFlag() const { return getBit(19); }
      bool l1aBxFIFOPartialFullFlag() const { return getBit(20); }
      bool feEventLengthFIFOPartialFullFlag() const { return getBit(21); }
      bool feFPGAPartialFullFlag() const { return getBit(22); }
      //bit 22 undefined
      bool qdrMemoryEmptyFlag() const { return getBit(24); }
      bool frameAddressFIFOEmptyFlag() const { return getBit(25); }
      bool totalLengthFIFOEmptyFlag() const { return getBit(26); }
      bool trackerHeaderFIFOEmptyFlag() const { return getBit(27); }
      bool l1aBxFIFOEmptyFlag() const { return getBit(28); }
      bool feEventLengthFIFOEmptyFlag() const { return getBit(29); }
      bool feFPGAEmptyFlag() const { return getBit(30); }
      //bit 31 undefined
      void print(std::ostream& os) const { printHex(&data_,4,os); }
      void printFlags(std::ostream& os) const;
    private:
      bool getBit(const uint8_t num) const { return ( (0x1<<num) & (data_) ); }
      const uint32_t data_;
    };

  class FEDFEHeader
    {
    public:
      //factory function: allocates new FEDFEHeader derrivative of appropriate type
      static std::auto_ptr<FEDFEHeader> newFEHeader(FEDHeaderType headerType, const uint8_t* headerBuffer);
      virtual ~FEDFEHeader();
      //the length of the header
      virtual size_t lengthInBytes() const = 0;
      //check that there are no errors indicated in which ever error bits are available in the header
      //check bits for both APVs on a channel
      bool checkChannelStatusBits(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return checkChannelStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      virtual bool checkChannelStatusBits(uint8_t internalFEDChannelNum) const = 0;
      //check bits for one APV
      bool checkStatusBits(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return checkStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
      virtual bool checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const = 0;
      virtual void print(std::ostream& os) const = 0;
    };

  class FEDAPVErrorHeader : public FEDFEHeader
    {
    public:
      FEDAPVErrorHeader(const uint8_t* headerBuffer);
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
      FEDFullDebugHeader(const uint8_t* headerBuffer);
      virtual ~FEDFullDebugHeader();
      virtual size_t lengthInBytes() const;
      virtual bool checkChannelStatusBits(uint8_t internalFEDChannelNum) const;
      virtual bool checkStatusBits(uint8_t internalFEDChannelNum, uint8_t apvNum) const;
      virtual void print(std::ostream& os) const;
  
      uint8_t feUnitMajorityAddress(uint8_t internalFEUnitNum) const { return feWord(internalFEUnitNum)[9]; }
      FEDBackendStatusRegister beStatusRegister() const { return FEDBackendStatusRegister(get32BitWordFrom(feWord(0)+10)); }
      uint32_t daqRegister() const { return get32BitWordFrom(feWord(7)+10); }
      uint32_t daqRegister2() const { return get32BitWordFrom(feWord(6)+10); }
      uint16_t feUnitLength(uint8_t internalFEUnitNum) const { return ( (feWord(internalFEUnitNum)[15]<<8) | (feWord(internalFEUnitNum)[14]) ); }
      bool fePresent(uint8_t internalFEUnitNum) const { return (feUnitLength(internalFEUnitNum) != 0); }
  
      //These methods return true if there was an error of the appropriate type (ie if the error bit is 0).
      //They return false if the error could not occur due to a more general error.
      //was channel unlocked
      bool unlocked(uint8_t internalFEDChannelNum) const { return unlockedFromBit(internalFEDChannelNum); }
      bool unlocked(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return unlocked(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //was channel out of sync if it was unlocked
      bool outOfSync(uint8_t internalFEDChannelNum) const
	{ return ( !unlocked(internalFEDChannelNum) && outOfSyncFromBit(internalFEDChannelNum) ); }
      bool outOfSync(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return outOfSync(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //was there an internal APV error if it was in sync
      bool apvError(uint8_t internalFEDChannelNum, uint8_t apvNum) const
	{
          return ( !unlockedFromBit(internalFEDChannelNum) &&
                   !outOfSyncFromBit(internalFEDChannelNum) &&
                   apvErrorFromBit(internalFEDChannelNum,apvNum) );
        }
      bool apvError(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvError(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
      //was the APV address wrong if it was in sync (does not depend on APV internal error bit)
      bool apvAddressError(uint8_t internalFEDChannelNum, uint8_t apvNum) const
	{
          return ( !unlockedFromBit(internalFEDChannelNum) &&
                   !outOfSyncFromBit(internalFEDChannelNum) &&
                   apvAddressErrorFromBit(internalFEDChannelNum,apvNum) );
        }
      bool apvAddressError(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvAddressError(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
  
      //These methods return true if there was an error of the appropriate type (ie if the error bit is 0).
      //They ignore any previous errors which make the status bits meaningless and return the value of the bit anyway.
      //In general, the methods above which only return an error for the likely cause are more useful.
      bool unlockedFromBit(uint8_t internalFEDChannelNum) const { return !getBitVal(internalFEDChannelNum,5); }
      bool unlockedFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return unlockedFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      bool outOfSyncFromBit(uint8_t internalFEDChannelNum) const { return !getBitVal(internalFEDChannelNum,4); }
      bool outOfSyncFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return outOfSyncFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      bool apvErrorFromBit(uint8_t internalFEDChannelNum, uint8_t apvNum) const { return !getBitVal(internalFEDChannelNum,0+2*apvNum); }
      bool apvErrorFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvErrorFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
      bool apvAddressErrorFromBit(uint8_t internalFEDChannelNum, uint8_t apvNum) const { return !getBitVal(internalFEDChannelNum,1+2*apvNum); }
      bool apvAddressErrorFromBit(uint8_t internalFEUnitNum, uint8_t internalChannelNum, uint8_t apvNum) const
	{ return apvAddressErrorFromBit(internalFEDChannelNum(internalFEUnitNum,internalChannelNum),apvNum); }
    private:
      bool getBitVal(uint8_t internalFEDChannelNum, uint8_t bit) const;
      static uint32_t get32BitWordFrom(const uint8_t* startOfWord);
      const uint8_t* feWord(uint8_t internalFEUnitNum) const { return header_+internalFEUnitNum*2*8; }
      static const size_t FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS = FEUNITS_PER_FED*2;
      static const size_t FULL_DEBUG_HEADER_SIZE_IN_BYTES = FULL_DEBUG_HEADER_SIZE_IN_64BIT_WORDS*8;
      uint8_t header_[FULL_DEBUG_HEADER_SIZE_IN_BYTES];
    };

  //
  // Inline function definitions
  //

  inline uint8_t internalFEDChannelNum(uint8_t internalFEUnitNum, uint8_t internalChannelNum)
    {
      return (internalFEUnitNum*FEDCH_PER_FEUNIT + internalChannelNum);
    }

  inline std::ostream& operator << (std::ostream& os, const FEDDAQHeader& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDDAQTrailer& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const TrackerSpecialHeader& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDStatusRegister& obj) { obj.print(os); return os; }
  inline std::ostream& operator << (std::ostream& os, const FEDFEHeader& obj) { obj.print(os); return os; }

  //TrackerSpecialHeader

  inline TrackerSpecialHeader::TrackerSpecialHeader()
    : wordSwapped_(false)
    {
    }

  inline bool TrackerSpecialHeader::majorityAddressErrorForFEUnit(uint8_t internalFEUnitNum) const
    {
      return ( (0x1<<internalFEUnitNum) & apvAddressErrorRegister() );
    }

  inline bool TrackerSpecialHeader::feEnabled(uint8_t internalFEUnitNum) const
    {
      return ( (0x1<<internalFEUnitNum) & feEnableRegister() );
    }

  inline bool TrackerSpecialHeader::feOverflow(uint8_t internalFEUnitNum) const
    {
      return ( (0x1<<internalFEUnitNum) & feOverflowRegister() );
    }

  inline uint16_t TrackerSpecialHeader::fedStatusRegisterWord() const
    {
      //get 16 bits
      uint16_t statusRegister = ( (specialHeader_[(FEDSTATUS+1)]<<8) | specialHeader_[FEDSTATUS]);
      return statusRegister;
    }

  //FEDFEHeader

  inline std::auto_ptr<FEDFEHeader> FEDFEHeader::newFEHeader(FEDHeaderType headerType, const uint8_t* headerBuffer)
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
      const uint8_t* pFEWord = feWord(internalFEDChannelNum / FEDCH_PER_FEUNIT);
      const uint8_t bitInFeWord = (internalFEDChannelNum % FEDCH_PER_FEUNIT) * 6 + bit;
      return ( pFEWord[bitInFeWord/8] & (0x1 << bitInFeWord%8) );
    }

  inline uint32_t FEDFullDebugHeader::get32BitWordFrom(const uint8_t* startOfWord)
    {
      return ( startOfWord[0] | (startOfWord[1]<<8) | (startOfWord[2]<<16) | (startOfWord[3]<<24) );
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
  
  inline void FEDDAQHeader::print(std::ostream& os) const
    {
      printHex(header_,8,os);
    }

  //FEDDAQTrailer

  inline FEDDAQTrailer::FEDDAQTrailer(const uint8_t* trailer)
    {
      memcpy(trailer_,trailer,8);
    }

  inline uint32_t FEDDAQTrailer::eventLengthIn64BitWords() const
    {
      return ( trailer_[4] | (trailer_[5]<<8) | (trailer_[6]<<16) );
    }

  inline uint16_t FEDDAQTrailer::crc() const
    {
      return ( trailer_[2] | (trailer_[3]<<8) );
    }

}

#endif //ndef EventFilter_SiStripRawToDigi_FEDBuffer_H
