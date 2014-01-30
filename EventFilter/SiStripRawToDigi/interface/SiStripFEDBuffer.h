#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H

#include "boost/cstdint.hpp"
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <cstring>
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

namespace sistrip {

  //
  // Class definitions
  //

  //class representing standard (non-spy channel) FED buffers
  class FEDBuffer final : public FEDBufferBase
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
      bool feGoodWithoutAPVEmulatorCheck(const uint8_t internalFEUnitNum) const;
      //check that a FE unit is present in the data.
      //The high order byte of the FEDStatus register in the tracker special header is used in APV error mode.
      //The FE length from the full debug header is used in full debug mode.
      bool fePresent(uint8_t internalFEUnitNum) const;
      //check that a channel is present in data, found, on a good FE unit and has no errors flagged in status bits
      virtual bool channelGood(const uint8_t internalFEDannelNum, const bool doAPVeCheck=true) const;

      //functions to check buffer. All return true if there is no problem.
      //minimum checks to do before using buffer
      virtual bool doChecks(bool doCRC=true) const;
  
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
      static void throwUnorderedData(const uint8_t currentStrip, const uint8_t firstStripOfNewCluster);
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

  //FEDBuffer

  inline const FEDFEHeader* FEDBuffer::feHeader() const
    {
      return feHeader_.get();
    }
  
  inline bool FEDBuffer::feGood(const uint8_t internalFEUnitNum) const
    {
      return ( !majorityAddressErrorForFEUnit(internalFEUnitNum) && !feOverflow(internalFEUnitNum) && fePresent(internalFEUnitNum) );
    }

  inline bool FEDBuffer::feGoodWithoutAPVEmulatorCheck(const uint8_t internalFEUnitNum) const
    {
      return ( !feOverflow(internalFEUnitNum) && fePresent(internalFEUnitNum) );
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
      if (channelPayloadLength_) readNewClusterInfo();
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
	if (hasData()) {
          const uint8_t oldStrip = currentStrip_;
          readNewClusterInfo();
          if ( !(currentStrip_ > oldStrip) ) throwUnorderedData(oldStrip,currentStrip_);
        }
      }
      return (*this);
    }
  
  inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ (int)
    {
      ++(*this); return *this;
    }
  
  inline void FEDZSChannelUnpacker::readNewClusterInfo()
    {
      currentStrip_ = data_[(currentOffset_++)^7];
      valuesLeftInCluster_ = data_[(currentOffset_++)^7]-1;
    }

}

#endif //ndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
