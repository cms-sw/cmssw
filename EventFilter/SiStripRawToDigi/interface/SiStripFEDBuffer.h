#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBuffer_H

#include "boost/cstdint.hpp"
#include <string>
#include <vector>
#include <memory>
#include <ostream>
#include <cstring>
#include <cmath>
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

namespace sistrip {
  constexpr uint16_t BITS_PER_BYTE = 8;

  //
  // Class definitions
  //

  //class representing standard (non-spy channel) FED buffers
  class FEDBuffer final : public FEDBufferBase
    {
    public:
      //construct from buffer
      //if allowBadBuffer is set to true then exceptions will not be thrown if the channel lengths do not make sense or the event format is not recognized
      FEDBuffer(const uint8_t* fedBuffer, const uint16_t fedBufferSize, const bool allowBadBuffer = false);
      ~FEDBuffer() override;
      void print(std::ostream& os) const override;
      const FEDFEHeader* feHeader() const;
      //check that a FE unit is enabled, has a good majority address and, if in full debug mode, that it is present
      bool feGood(const uint8_t internalFEUnitNum) const;
      bool feGoodWithoutAPVEmulatorCheck(const uint8_t internalFEUnitNum) const;
      //check that a FE unit is present in the data.
      //The high order byte of the FEDStatus register in the tracker special header is used in APV error mode.
      //The FE length from the full debug header is used in full debug mode.
      bool fePresent(uint8_t internalFEUnitNum) const;
      //check that a channel is present in data, found, on a good FE unit and has no errors flagged in status bits
      using sistrip::FEDBufferBase::channelGood;
      virtual bool channelGood(const uint8_t internalFEDannelNum, const bool doAPVeCheck=true) const;
      void setLegacyMode(bool legacy) { legacyUnpacker_ = legacy;}

      //functions to check buffer. All return true if there is no problem.
      //minimum checks to do before using buffer
      using sistrip::FEDBufferBase::doChecks;
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
      std::string checkSummary() const override;
    private:
      uint8_t nFEUnitsPresent() const;
      void findChannels();
      inline uint8_t getCorrectPacketCode() const { return packetCode(legacyUnpacker_); }
      uint16_t calculateFEUnitLength(const uint8_t internalFEUnitNumber) const;
      std::unique_ptr<FEDFEHeader> feHeader_;
      const uint8_t* payloadPointer_;
      uint16_t payloadLength_;
      uint8_t validChannels_;
      bool fePresent_[FEUNITS_PER_FED];
      bool legacyUnpacker_=false;
    };

  //class for unpacking data from ZS FED channels
  class FEDZSChannelUnpacker
    {
    public:
      static FEDZSChannelUnpacker zeroSuppressedModeUnpacker(const FEDChannel& channel);
      static FEDZSChannelUnpacker zeroSuppressedLiteModeUnpacker(const FEDChannel& channel);
      static FEDZSChannelUnpacker preMixRawModeUnpacker(const FEDChannel& channel);
      FEDZSChannelUnpacker();
      uint8_t sampleNumber() const;
      uint8_t adc() const;
      uint16_t adcPreMix() const;
      bool hasData() const;
      FEDZSChannelUnpacker& operator ++ ();
      FEDZSChannelUnpacker& operator ++ (int);
    private:
      //pointer to beginning of FED or FE data, offset of start of channel payload in data and length of channel payload
      FEDZSChannelUnpacker(const uint8_t* payload, const uint16_t channelPayloadOffset, const int16_t channelPayloadLength, const uint16_t offsetIncrement=1);
      void readNewClusterInfo();
      static void throwBadChannelLength(const uint16_t length);
      void throwBadClusterLength();
      static void throwUnorderedData(const uint8_t currentStrip, const uint8_t firstStripOfNewCluster);
      const uint8_t* data_;
      uint16_t currentOffset_;
      uint16_t offsetIncrement_;
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
      uint16_t currentOffset_;
      uint8_t currentStrip_;
      uint16_t valuesLeft_;
    };

  //class for unpacking data from any FED channels with a non-integer words bits stripping mode
  class FEDBSChannelUnpacker
    {
    public:
      static FEDBSChannelUnpacker virginRawModeUnpacker(const FEDChannel& channel, uint16_t num_bits);
      static FEDBSChannelUnpacker zeroSuppressedModeUnpacker(const FEDChannel& channel, uint16_t num_bits);
      static FEDBSChannelUnpacker zeroSuppressedLiteModeUnpacker(const FEDChannel& channel, uint16_t num_bits);
      FEDBSChannelUnpacker();
      uint8_t sampleNumber() const;
      uint16_t adc() const;
      bool hasData() const;
      FEDBSChannelUnpacker& operator ++ ();
      FEDBSChannelUnpacker& operator ++ (int);
    private:
      //pointer to beginning of FED or FE data, offset of start of channel payload in data and length of channel payload
      FEDBSChannelUnpacker(const uint8_t* payload, const uint16_t channelPayloadOffset, const int16_t channelPayloadLength, const uint16_t offsetIncrement, bool useZS);
      void readNewClusterInfo();
      static void throwBadChannelLength(const uint16_t length);
      static void throwBadWordLength(const uint16_t word_length);
      static void throwUnorderedData(const uint8_t currentStrip, const uint8_t firstStripOfNewCluster);
      const uint8_t* data_;
      uint16_t oldWordOffset_;
      uint16_t currentWordOffset_;
      uint16_t currentLocalBitOffset_;
      uint16_t bitOffsetIncrement_;
      uint8_t currentStrip_;
      uint16_t channelPayloadOffset_;
      uint16_t channelPayloadLength_;
      bool useZS_;
      uint8_t valuesLeftInCluster_;
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

  //FEDBSChannelUnpacker

  inline FEDBSChannelUnpacker::FEDBSChannelUnpacker()
    : data_(nullptr),
      oldWordOffset_(0), currentWordOffset_(0),
      currentLocalBitOffset_(0),
      bitOffsetIncrement_(10),
      currentStrip_(0),
      channelPayloadOffset_(0), channelPayloadLength_(0),
      useZS_(false), valuesLeftInCluster_(0)
    { }

  inline FEDBSChannelUnpacker::FEDBSChannelUnpacker(const uint8_t* payload, const uint16_t channelPayloadOffset, const int16_t channelPayloadLength, const uint16_t offsetIncrement, bool useZS)
    : data_(payload),
      oldWordOffset_(0), currentWordOffset_(channelPayloadOffset),
      currentLocalBitOffset_(0),
      bitOffsetIncrement_(offsetIncrement),
      currentStrip_(0),
      channelPayloadOffset_(channelPayloadOffset),
      channelPayloadLength_(channelPayloadLength),
      useZS_(useZS), valuesLeftInCluster_(0)
    {
      if (bitOffsetIncrement_>16) throwBadWordLength(bitOffsetIncrement_); // more than 2 words... still to be implemented
      if (useZS_ && channelPayloadLength_) readNewClusterInfo();
    }

  inline FEDBSChannelUnpacker FEDBSChannelUnpacker::virginRawModeUnpacker(const FEDChannel& channel, uint16_t num_bits)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      if (num_bits<=0 or num_bits>16) throwBadWordLength(num_bits);
      FEDBSChannelUnpacker result(channel.data(), channel.offset()+3, length-3, num_bits, false);
      return result;
    }

  inline FEDBSChannelUnpacker FEDBSChannelUnpacker::zeroSuppressedModeUnpacker(const FEDChannel& channel, uint16_t num_bits)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDBSChannelUnpacker result(channel.data(), channel.offset()+7, length-7, num_bits, true);
      return result;
    }

  inline FEDBSChannelUnpacker FEDBSChannelUnpacker::zeroSuppressedLiteModeUnpacker(const FEDChannel& channel, uint16_t num_bits)
    {
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDBSChannelUnpacker result(channel.data(), channel.offset()+2, length-2, num_bits, true);
      return result;
    }

  inline uint8_t FEDBSChannelUnpacker::sampleNumber() const
    {
      return currentStrip_;
    }

  inline uint16_t FEDBSChannelUnpacker::adc() const
    {
      uint16_t bits_missing = (bitOffsetIncrement_-BITS_PER_BYTE)+currentLocalBitOffset_;
      uint16_t adc = (data_[currentWordOffset_^7]<<bits_missing);
      if (currentWordOffset_>oldWordOffset_) {
        adc += ( (data_[(currentWordOffset_+1)^7]>>(BITS_PER_BYTE-bits_missing)) );
      }
      return (adc&((1<<bitOffsetIncrement_)-1));
    }

  inline bool FEDBSChannelUnpacker::hasData() const
    {
      const uint16_t nextChanWordOffset = channelPayloadOffset_+channelPayloadLength_;
      if ( currentWordOffset_ + 1 < nextChanWordOffset ) {
        return true; // fast case: 2 bytes always fit an ADC (even if offset)
      } else { // close to end
        const uint16_t plusOneBitOffset = currentLocalBitOffset_+bitOffsetIncrement_;
        const uint16_t plusOneWordOffset = currentWordOffset_ + plusOneBitOffset/BITS_PER_BYTE;
        return ( plusOneBitOffset % BITS_PER_BYTE ) ? ( plusOneWordOffset < nextChanWordOffset ) : ( plusOneWordOffset <= nextChanWordOffset );
      }
    }

  inline FEDBSChannelUnpacker& FEDBSChannelUnpacker::operator ++ ()
    {
      oldWordOffset_ = currentWordOffset_;
      currentLocalBitOffset_ += bitOffsetIncrement_;
      while (currentLocalBitOffset_>=BITS_PER_BYTE) {
        currentWordOffset_++;
        currentLocalBitOffset_ -= BITS_PER_BYTE;
      }
      if (useZS_) {
	if (valuesLeftInCluster_) { currentStrip_++; valuesLeftInCluster_--; }
	else {
	  if (hasData()) {
	    const uint8_t oldStrip = currentStrip_;
	    readNewClusterInfo();
	    if ( !(currentStrip_ > oldStrip) ) throwUnorderedData(oldStrip,currentStrip_);
	  }
	}
      } else { currentStrip_++; }
      return (*this);
    }

  inline FEDBSChannelUnpacker& FEDBSChannelUnpacker::operator ++ (int)
    {
      ++(*this); return *this;
    }

  inline void FEDBSChannelUnpacker::readNewClusterInfo()
    {
      if ( currentLocalBitOffset_ ) {
        ++currentWordOffset_;
        currentLocalBitOffset_ = 0;
      }
      currentStrip_ = data_[(currentWordOffset_++)^7];
      valuesLeftInCluster_ = data_[(currentWordOffset_++)^7]-1;
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
    : data_(nullptr),
      offsetIncrement_(1),
      valuesLeftInCluster_(0),
      channelPayloadOffset_(0),
      channelPayloadLength_(0)
    { }

  inline FEDZSChannelUnpacker::FEDZSChannelUnpacker(const uint8_t* payload, const uint16_t channelPayloadOffset, const int16_t channelPayloadLength, const uint16_t offsetIncrement)
    : data_(payload),
      currentOffset_(channelPayloadOffset),
      offsetIncrement_(offsetIncrement),
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
  
  inline FEDZSChannelUnpacker FEDZSChannelUnpacker::preMixRawModeUnpacker(const FEDChannel& channel)
    {
      //CAMM - to modify more ?
      uint16_t length = channel.length();
      if (length & 0xF000) throwBadChannelLength(length);
      FEDZSChannelUnpacker result(channel.data(),channel.offset()+7,length-7,2);
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
  
  inline uint16_t FEDZSChannelUnpacker::adcPreMix() const
    {
      return ( data_[currentOffset_^7] + ((data_[(currentOffset_+1)^7]&0x03)<<8) );
    }
  
  inline bool FEDZSChannelUnpacker::hasData() const
    {
      return (currentOffset_<channelPayloadOffset_+channelPayloadLength_);
    }
  
  inline FEDZSChannelUnpacker& FEDZSChannelUnpacker::operator ++ ()
    {
      if (valuesLeftInCluster_) {
	currentStrip_++;
	currentOffset_ += offsetIncrement_;
        valuesLeftInCluster_--;
      } else {
	currentOffset_ += offsetIncrement_;
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
