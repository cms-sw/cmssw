#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H

#include "boost/cstdint.hpp"
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <cstring>
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

namespace sistrip {

  //
  // Class definitions
  //
  
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

  //class representing standard (non-spy channel) FED buffers
  class FEDBuffer : public FEDBufferBase
    {
    public:
      //construct from buffer
      //if allowBadBuffer is set to true then exceptions will not be thrown if the channel lengths do not make sense or the event format is not recognized
      FEDBuffer(const uint8_t* fedBuffer, const size_t fedBufferSize, const bool allowBadBuffer = false);
      virtual ~FEDBuffer();
      virtual void print(std::ostream& os) const;
      const FEDFEHeader* feHeader() const;
      //check that a FE unit is enabled, has a good majority address and, if in full debug mode, that it is present
      bool feGood(const uint8_t internalFEUnitNum) const;
      //check that a FE unit is present in the data.
      //The high order byte of the FEDStatus register in the tracker special header is used in APV error mode.
      //The FE length from the full debug header is used in full debug mode.
      bool fePresent(uint8_t internalFEUnitNum) const;
      //check that a channel is present in data, found, on a good FE unit and has no errors flagged in status bits
      virtual bool channelGood(const uint8_t internalFEDannelNum) const;

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
      bool checkStatusBits(const uint8_t internalFEDChannelNum) const;
      bool checkStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const;
      //same but for all channels on enabled FE units
      bool checkAllChannelStatusBits() const;
      
      //check that all FE unit payloads are present
      bool checkFEPayloadsPresent() const;
  
      //print a summary of all checks
      virtual std::string checkSummary() const;
    private:
      uint8_t nFEUnitsPresent() const;
      void findChannels();
      uint8_t getCorrectPacketCode() const;
      uint16_t calculateFEUnitLength(const uint8_t internalFEUnitNumber) const;
      std::auto_ptr<FEDFEHeader> feHeader_;
      const uint8_t* payloadPointer_;
      uint16_t payloadLength_;
      uint8_t validChannels_;
      bool fePresent_[FEUNITS_PER_FED];
    };

  //class for unpacking data from ZS FED channels
  class FEDZSChannelUnpacker
    {
    public:
      static FEDZSChannelUnpacker zeroSuppressedModeUnpacker(const FEDChannel& channel);
      static FEDZSChannelUnpacker zeroSuppressedLiteModeUnpacker(const FEDChannel& channel);
      FEDZSChannelUnpacker();
      uint8_t sampleNumber() const;
      uint8_t adc() const;
      bool hasData() const;
      FEDZSChannelUnpacker& operator ++ ();
      FEDZSChannelUnpacker& operator ++ (int);
    private:
      //pointer to begining of FED or FE data, offset of start of channel payload in data and length of channel payload
      FEDZSChannelUnpacker(const uint8_t* payload, const size_t channelPayloadOffset, const int16_t channelPayloadLength);
      void readNewClusterInfo();
      static void throwBadChannelLength(const uint16_t length);
      void throwBadClusterLength();
      const uint8_t* data_;
      size_t currentOffset_;
      uint8_t currentStrip_;
      uint8_t valuesLeftInCluster_;
      uint16_t channelPayloadOffset_;
      uint16_t channelPayloadLength_;
    };

  //class for unpacking data from raw FED channels
  class FEDRawChannelUnpacker
    {
    public:
      static FEDRawChannelUnpacker scopeModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      static FEDRawChannelUnpacker virginRawModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      static FEDRawChannelUnpacker procRawModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      explicit FEDRawChannelUnpacker(const FEDChannel& channel);
      uint8_t sampleNumber() const;
      uint16_t adc() const;
      bool hasData() const;
      FEDRawChannelUnpacker& operator ++ ();
      FEDRawChannelUnpacker& operator ++ (int);
    private:
      static void throwBadChannelLength(const uint16_t length);
      const uint8_t* data_;
      size_t currentOffset_;
      uint8_t currentStrip_;
      uint16_t valuesLeft_;
    };

  //
  // Inline function definitions
  //

  inline std::ostream& operator << (std::ostream& os, const FEDBufferBase& obj) { obj.print(os); os << obj.checkSummary(); return os; }

  //FEDBuffer

  inline const FEDFEHeader* FEDBuffer::feHeader() const
    {
      return feHeader_.get();
    }
  
  inline bool FEDBuffer::feGood(const uint8_t internalFEUnitNum) const
    {
      return ( !majorityAddressErrorForFEUnit(internalFEUnitNum) && !feOverflow(internalFEUnitNum) && fePresent(internalFEUnitNum) );
    }
  
  inline bool FEDBuffer::fePresent(uint8_t internalFEUnitNum) const
    {
      return fePresent_[internalFEUnitNum];
    }
  
  inline bool FEDBuffer::checkStatusBits(const uint8_t internalFEDChannelNum) const
    {
      return feHeader_->checkChannelStatusBits(internalFEDChannelNum);
    }
  
  inline bool FEDBuffer::checkStatusBits(const uint8_t internalFEUnitNum, const uint8_t internalChannelNum) const
    {
      return checkStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum));
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
	       checkAPVEAddressValid() &&
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

  //FEDRawChannelUnpacker

  inline FEDRawChannelUnpacker::FEDRawChannelUnpacker(const FEDChannel& channel)
    : data_(channel.data()),
      currentOffset_(channel.offset()+3),
      currentStrip_(0),
      valuesLeft_((channel.length()-3)/2)
    {
      if ((channel.length()-3)%2) throwBadChannelLength(channel.length());
    }
  
  inline uint8_t FEDRawChannelUnpacker::sampleNumber() const
    {
      return currentStrip_;
    }
  
  inline uint16_t FEDRawChannelUnpacker::adc() const
    {
      return ( data_[currentOffset_^7] + ((data_[(currentOffset_+1)^7]&0x03)<<8) );
    }
  
  inline bool FEDRawChannelUnpacker::hasData() const
    {
      return valuesLeft_;
    }
  
  inline FEDRawChannelUnpacker& FEDRawChannelUnpacker::operator ++ ()
    {
      currentOffset_ += 2;
      currentStrip_++;
      valuesLeft_--;
      return (*this);
    }
  
  inline FEDRawChannelUnpacker& FEDRawChannelUnpacker::operator ++ (int)
    {
      ++(*this); return *this;
    }

  //FEDZSChannelUnpacker
  
  inline FEDZSChannelUnpacker::FEDZSChannelUnpacker()
    : data_(NULL),
      valuesLeftInCluster_(0),
      channelPayloadOffset_(0),
      channelPayloadLength_(0)
    { }

  inline FEDZSChannelUnpacker::FEDZSChannelUnpacker(const uint8_t* payload, const size_t channelPayloadOffset, const int16_t channelPayloadLength)
    : data_(payload),
      currentOffset_(channelPayloadOffset),
      currentStrip_(0),
      valuesLeftInCluster_(0),
      channelPayloadOffset_(channelPayloadOffset),
      channelPayloadLength_(channelPayloadLength)
    {
      readNewClusterInfo();
    }
  
  inline FEDZSChannelUnpacker FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(const FEDChannel& channel)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDZSChannelUnpacker result(channel.data(),channel.offset()+7,length-7);
      return result;
    }

  inline FEDZSChannelUnpacker FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(const FEDChannel& channel)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDZSChannelUnpacker result(channel.data(),channel.offset()+2,length-2);
      return result;
    }
  
  inline uint8_t FEDZSChannelUnpacker::sampleNumber() const
    {
      return currentStrip_;
    }
  
  inline uint8_t FEDZSChannelUnpacker::adc() const
    {
      return data_[currentOffset_^7];
    }
  
  inline bool FEDZSChannelUnpacker::hasData() const
    {
      return (currentOffset_<channelPayloadOffset_+channelPayloadLength_);
    }
  
  inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ ()
    {
      if (valuesLeftInCluster_) {
	currentStrip_++;
	currentOffset_++;
        valuesLeftInCluster_--;
      } else {
	currentOffset_++;
	readNewClusterInfo();
      }
      return (*this);
    }
  
  inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ (int)
    {
      ++(*this); return *this;
    }
  
  inline void FEDZSChannelUnpacker::readNewClusterInfo()
    {
      if (channelPayloadLength_) {
	currentStrip_ = data_[(currentOffset_++)^7];
	valuesLeftInCluster_ = data_[(currentOffset_++)^7]-1;
      }
    }

}

#endif //ndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
