#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H

#include "boost/cstdint.hpp"
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include <iostream>

namespace sistrip {

  //
  // Class definitions
  //

  class FEDBufferBase
    {
    public:
      FEDBufferBase(const uint8_t* fedBuffer, size_t fedBufferSize, bool allowUnrecognizedFormat = false);
      virtual ~FEDBufferBase();
      //dump buffer to stream
      void dump(std::ostream& os) const { printHex(orderedBuffer_,bufferSize_,os); }
      //dump original buffer before word swapping
      void dumpOriginalBuffer(std::ostream& os) const { printHex(originalBuffer_,bufferSize_,os); }
      virtual void print(std::ostream& os) const;
      //calculate the CRC from the buffer
      uint16_t calcCRC() const;
  
      //methods to get parts of the buffer
      FEDDAQHeader daqHeader() const { return daqHeader_; }
      FEDDAQTrailer daqTrailer() const { return daqTrailer_; }
      size_t bufferSize() const { return bufferSize_; }
      TrackerSpecialHeader trackerSpecialHeader() const { return specialHeader_; }
      //methods to get info from DAQ header
      FEDDAQEventType daqEventType() const { return daqHeader_.eventType(); }
      uint32_t daqLvl1ID() const { return daqHeader_.l1ID(); }
      uint16_t daqBXID() const { return daqHeader_.bxID(); }
      uint16_t daqSourceID() const { return daqHeader_.sourceID(); }
      uint16_t sourceID() const { return daqSourceID(); }
      //methods to get info from DAQ trailer
      uint32_t daqEventLengthIn64bitWords() const { return daqTrailer_.eventLengthIn64BitWords(); }
      uint32_t daqEventLengthInBytes() const { return daqTrailer_.eventLengthInBytes(); }
      uint16_t daqCRC() const { return daqTrailer_.crc(); }
      FEDTTSBits daqTTSState() const { return daqTrailer_.ttsBits(); }
      //methods to get info from the tracker special header
      FEDBufferFormat bufferFormat() const { return specialHeader_.bufferFormat(); }
      FEDHeaderType headerType() const { return specialHeader_.headerType(); }
      FEDReadoutMode readoutMode() const { return specialHeader_.readoutMode(); }
      FEDDataType dataType() const { return specialHeader_.dataType(); }
      uint8_t apveAddress() const { return specialHeader_.apveAddress(); }
      bool majorityAddressErrorForFEUnit(uint8_t internalFEUnitNum) const { return specialHeader_.majorityAddressErrorForFEUnit(internalFEUnitNum); }
      bool feEnabled(uint8_t internalFEUnitNum) const { return specialHeader_.feEnabled(internalFEUnitNum); }
      uint8_t nFEUnitsEnabled() const;
      bool feOverflow(uint8_t internalFEUnitNum) const { return specialHeader_.feOverflow(internalFEUnitNum); }
      FEDStatusRegister fedStatusRegister() const { return specialHeader_.fedStatusRegister(); }
  
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
      bool checkCRC() const { return ( checkNoSlinkCRCError() && (calcCRC()==daqCRC()) ); }
      bool checkMajorityAddresses() const;
      //methods to check tracker special header
      bool checkBufferFormat() const { return (bufferFormat() != BUFFER_FORMAT_INVALID); }
      bool checkHeaderType() const { return (headerType() != HEADER_TYPE_INVALID); }
      bool checkReadoutMode() const { return (readoutMode() != READOUT_MODE_INVALID); }
      bool checkAPVEAddressValid() const { return (apveAddress() <= APV_MAX_ADDRESS); }
      bool checkNoFEOverflows() const { return !specialHeader_.feOverflowRegister(); }
      //methods to check daq header and trailer
      bool checkNoSlinkCRCError() const { return !daqTrailer_.slinkCRCError(); }
      bool checkNoSLinkTransmissionError() const { return !daqTrailer_.slinkTransmissionError(); }
      bool checkSourceIDs() const;
      bool checkNoUnexpectedSourceID() const { return !daqTrailer_.badFEDID(); }
      bool checkNoExtraHeadersOrTrailers() const { return ( (daqHeader_.boeNibble() == 0x5) && (daqTrailer_.eoeNibble() == 0xA) ); }
      bool checkLengthFromTrailer() const { return (bufferSize() == daqEventLengthInBytes()); }
    protected:
      const uint8_t* getPointerToDataAfterTrackerSpecialHeader() const
	{ return orderedBuffer_+16; }
      uint8_t* getPointerToDataAfterTrackerSpecialHeader();
      const uint8_t* getPointerToByteAfterEndOfPayload() const
	{ return orderedBuffer_+bufferSize_-8; }
      uint8_t* getPointerToByteAfterEndOfPayload();
    private:
      const uint8_t* originalBuffer_;
      const uint8_t* orderedBuffer_;
      const size_t bufferSize_;
      FEDDAQHeader daqHeader_;
      FEDDAQTrailer daqTrailer_;
      TrackerSpecialHeader specialHeader_;
    };

  class FEDZSChannelUnpacker;
  class FEDRawChannelUnpacker;

  class FEDChannel
    {
    public:
      FEDChannel(const uint8_t* data, size_t offset);
      uint16_t length() const { return length_; }
      uint8_t packetCode() const { return data_[(offset_+2)^7]; }
    private:
      friend class FEDBuffer;
      friend class FEDZSChannelUnpacker;
      friend class FEDRawChannelUnpacker;
      const uint8_t* data() const { return data_; }
      size_t offset() const { return offset_; }
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
      virtual void print(std::ostream& os) const;
      const FEDFEHeader* feHeader() const { return feHeader_.get(); }
      //check that a FE unit is enabled, has a good majority address and, if in full debug mode, that it is present
      bool feGood(uint8_t internalFEUnitNum) const { return ( !majorityAddressErrorForFEUnit(internalFEUnitNum) && !feOverflow(internalFEUnitNum) && fePresent(internalFEUnitNum) ); }
      //check that a FE unit is present in the data.
      //The high order byte of the FEDStatus register in the tracker special header is used in APV error mode.
      //The FE length from the full debug header is used in full debug mode.
      bool fePresent(uint8_t internalFEUnitNum) const { return fePresent_[internalFEUnitNum]; }
      //check that channel is on enabled FE Unit and has no errors
      bool channelGood(uint8_t internalFEDChannelNum) const;
      bool channelGood(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return channelGood(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
      //return channel object for channel
      const FEDChannel& channel(uint8_t internalFEDChannelNum) const { return channels_[internalFEDChannelNum]; }
      const FEDChannel& channel(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
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
      bool checkStatusBits(uint8_t internalFEDChannelNum) const { return feHeader_->checkChannelStatusBits(internalFEDChannelNum); }
      bool checkStatusBits(uint8_t internalFEUnitNum, uint8_t internalChannelNum) const
	{ return checkStatusBits(internalFEDChannelNum(internalFEUnitNum,internalChannelNum)); }
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
      uint16_t calculateFEUnitLength(uint8_t internalFEUnitNumber) const;
      std::vector<FEDChannel> channels_;
      std::auto_ptr<FEDFEHeader> feHeader_;
      uint8_t* payloadPointer_;
      uint16_t payloadLength_;
      uint8_t lastValidChannel_;
      bool fePresent_[FEUNITS_PER_FED];
    };

  class FEDZSChannelUnpacker
    {
    public:
      static FEDZSChannelUnpacker zeroSuppressedModeUnpacker(const FEDChannel& channel);
      static FEDZSChannelUnpacker zeroSuppressedLiteModeUnpacker(const FEDChannel& channel);
      FEDZSChannelUnpacker() : data_(NULL), valuesLeftInCluster_(0), channelPayloadOffset_(0), channelPayloadLength_(0) { }
      uint8_t strip() const { return currentStrip_; }
      uint8_t adc() const { return data_[currentOffset_^7]; }
      bool hasData() const { return (currentOffset_<channelPayloadOffset_+channelPayloadLength_) ; }
      FEDZSChannelUnpacker& operator ++ ();
      FEDZSChannelUnpacker& operator ++ (int) { ++(*this); return *this; }
    private:
      //pointer to begining of FED or FE data, offset of start of channel payload in data and length of channel payload
      FEDZSChannelUnpacker(const uint8_t* payload, size_t channelPayloadOffset,int16_t channelPayloadLength);
      void readNewClusterInfo();
      static void throwBadChannelLength(uint16_t length);
      void throwBadClusterLength();
      const uint8_t* data_;
      size_t currentOffset_;
      uint8_t currentStrip_;
      uint8_t valuesLeftInCluster_;
      uint16_t channelPayloadOffset_;
      uint16_t channelPayloadLength_;
    };

  class FEDRawChannelUnpacker
    {
    public:
      static FEDRawChannelUnpacker scopeModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      static FEDRawChannelUnpacker virginRawModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      static FEDRawChannelUnpacker procRawModeUnpacker(const FEDChannel& channel) { return FEDRawChannelUnpacker(channel); }
      FEDRawChannelUnpacker(const FEDChannel& channel);
      uint8_t strip() const { return currentStrip_; }
      uint16_t adc() const { return ( data_[currentOffset_^7] + ((data_[(currentOffset_+1)^7]&0x03)<<8) ); }
      bool hasData() const { return valuesLeft_; }
      FEDRawChannelUnpacker& operator ++ ();
      FEDRawChannelUnpacker& operator ++ (int) { ++(*this); return *this; }
    private:
      static void throwBadChannelLength(uint16_t length);
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

  inline bool FEDBuffer::channelGood(uint8_t internalFEDChannelNum) const
    {
      return ( (internalFEDChannelNum <= lastValidChannel_) &&
	       feGood(internalFEDChannelNum/FEDCH_PER_FEUNIT) &&
	       checkStatusBits(internalFEDChannelNum) );
    }

  //FEDBufferBase

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

  //FEDChannel

  inline FEDChannel::FEDChannel(const uint8_t* data, size_t offset)
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

  inline FEDZSChannelUnpacker::FEDZSChannelUnpacker(const uint8_t* payload, size_t channelPayloadOffset,int16_t channelPayloadLength)
    : data_(payload),
    currentOffset_(channelPayloadOffset),
    currentStrip_(0),
    valuesLeftInCluster_(0),
    channelPayloadOffset_(channelPayloadOffset),
    channelPayloadLength_(channelPayloadLength)
    {
      readNewClusterInfo();
    }

  inline void FEDZSChannelUnpacker::readNewClusterInfo()
    {
      if (channelPayloadLength_) {
	currentStrip_ = data_[(currentOffset_++)^7];
	valuesLeftInCluster_ = data_[(currentOffset_++)^7]-1;
      }
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

}

#endif //ndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
