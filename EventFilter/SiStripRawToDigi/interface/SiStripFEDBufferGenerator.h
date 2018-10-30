#ifndef EventFilter_SiStripRawToDigi_SiStripFEDBufferGenerator_H
#define EventFilter_SiStripRawToDigi_SiStripFEDBufferGenerator_H

#include "boost/cstdint.hpp"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include <vector>
#include <list>
#include <utility>
#include <memory>

namespace sistrip {
  
  //
  // Class definitions
  //
  
  class FEDStripData
    {
    public:
      //class used to represent channel data
      class ChannelData {
      public:
        ChannelData(bool dataIsAlreadyConvertedTo8Bit, const size_t numberOfSamples,
                    const std::pair<uint16_t,uint16_t> medians = std::make_pair<uint16_t>(0,0));
        //number of samples
        size_t size() const;
        //get common mode medians for first and second APV
        std::pair<uint16_t,uint16_t> getMedians() const;
        //set common mode medians for first and second APV
        void setMedians(const std::pair<uint16_t,uint16_t> values);
        //get the 10bit value to be used for raw modes
        uint16_t getSample(const uint16_t sampleNumber) const;
        //get the 8 bit value to be used for ZS modes, converting it as the FED does if specified in constructor
        uint8_t get8BitSample(const uint16_t sampleNumber, uint16_t nBotBitsToDrop) const;
        uint16_t get10BitSample(const uint16_t sampleNumber) const;
        void setSample(const uint16_t sampleNumber, const uint16_t adcValue);
        //setting value directly is equivalent to get and set Sample but without length check
        uint16_t& operator [] (const size_t sampleNumber);
        const uint16_t& operator [] (const size_t sampleNumber) const;
      private:
        std::pair<uint16_t,uint16_t> medians_;
        std::vector<uint16_t> data_;
        bool dataIs8Bit_;
      };
      
      FEDStripData(const std::vector<ChannelData>& data);
      //specify whether the data is already in the 8bit ZS format (only affects what is returned by ChannelData::get8BitSample() if the value if >253)
      //if the data is for scope mode then specify the scope length
      FEDStripData(bool dataIsAlreadyConvertedTo8Bit = true, const size_t samplesPerChannel = STRIPS_PER_FEDCH);
      //access to elements
      ChannelData& operator [] (const uint8_t internalFEDChannelNum);
      const ChannelData& operator [] (const uint8_t internalFEDChannelNum) const;
      ChannelData& channel(const uint8_t internalFEDChannelNum);
      const ChannelData& channel(const uint8_t internalFEDChannelNum) const;
    private:
      std::vector<ChannelData> data_;
    };
  
  class FEDBufferPayload
    {
    public:
      FEDBufferPayload(const std::vector< std::vector<uint8_t> >& channelBuffers);
      //size of payload in bytes
      size_t lengthInBytes() const;
      //returns NULL if payload size is 0, otherwise return a pointer to the payload buffer
      const uint8_t* data() const;
      //size of FE unit payload
      uint16_t getFELength(const uint8_t internalFEUnitNum) const;
    private:
      void appendToBuffer(size_t* pIndexInBuffer, const uint8_t value);
      void appendToBuffer(size_t* pIndexInBuffer, std::vector<uint8_t>::const_iterator start, std::vector<uint8_t>::const_iterator finish);
      std::vector<uint8_t> data_;
      std::vector<uint16_t> feLengths_;
    };
  
  class FEDBufferPayloadCreator
    {
    public:
      //specify which FE units and channels should have data generated for them
      //If an FE unit is disabled then the channel is as well. The whole FE payload will be missing. 
      //If a channel is disabled then it is considered to have all zeros in the data but will be present in the data
      FEDBufferPayloadCreator(const std::vector<bool>& enabledFEUnits, const std::vector<bool>& enabledChannels);
      //create the payload for a particular mode
      FEDBufferPayload createPayload(FEDReadoutMode mode, uint8_t packetCode, const FEDStripData& data) const;
      FEDBufferPayload operator () (FEDReadoutMode mode, uint8_t packetCode, const FEDStripData& data) const;
    private:
      //fill vector with channel data
      void fillChannelBuffer(std::vector<uint8_t>* channelBuffer, FEDReadoutMode mode, uint8_t packetCode,
                             const FEDStripData::ChannelData& data, const bool channelEnabled) const;
      //fill the vector with channel data for raw mode
      void fillRawChannelBuffer(std::vector<uint8_t>* channelBuffer, const uint8_t packetCode,
                                const FEDStripData::ChannelData& data, const bool channelEnabled, const bool reorderData) const;
      //fill the vector with channel data for zero suppressed modes
      void fillZeroSuppressedChannelBuffer(std::vector<uint8_t>* channelBuffer, const uint8_t packetCode, const FEDStripData::ChannelData& data, const bool channelEnabled) const;
      void fillZeroSuppressedLiteChannelBuffer(std::vector<uint8_t>* channelBuffer, const FEDStripData::ChannelData& data, const bool channelEnabled, const FEDReadoutMode mode) const;
      void fillPreMixRawChannelBuffer(std::vector<uint8_t>* channelBuffer, const FEDStripData::ChannelData& data, const bool channelEnabled) const;
     //add the ZS cluster data for the channel to the end of the vector
      void fillClusterData(std::vector<uint8_t>* channelBuffer, uint8_t packetCode, const FEDStripData::ChannelData& data, const FEDReadoutMode mode) const;
      void fillClusterDataPreMixMode(std::vector<uint8_t>* channelBuffer, const FEDStripData::ChannelData& data) const;
      std::vector<bool> feUnitsEnabled_;
      std::vector<bool> channelsEnabled_;
    };
  
  class FEDBufferGenerator
    {
    public:
      //constructor in which you can specify the defaults for some parameters
      FEDBufferGenerator(const uint32_t l1ID = 0,
                         const uint16_t bxID = 0,
                         const std::vector<bool>& feUnitsEnabled = std::vector<bool>(FEUNITS_PER_FED,true),
                         const std::vector<bool>& channelsEnabled = std::vector<bool>(FEDCH_PER_FED,true),
                         const FEDReadoutMode readoutMode = READOUT_MODE_ZERO_SUPPRESSED,
                         const FEDHeaderType headerType = HEADER_TYPE_FULL_DEBUG,
                         const FEDBufferFormat bufferFormat = BUFFER_FORMAT_OLD_SLINK,
                         const FEDDAQEventType evtType = DAQ_EVENT_TYPE_PHYSICS);
      //methods to get and set the defaults
      uint32_t getL1ID() const;
      uint16_t getBXID() const;
      FEDReadoutMode getReadoutMode() const;
      FEDHeaderType getHeaderType() const;
      FEDBufferFormat getBufferFormat() const;
      FEDDAQEventType getDAQEventType() const;
      FEDBufferGenerator& setL1ID(const uint32_t newL1ID);
      FEDBufferGenerator& setBXID(const uint16_t newBXID);
      FEDBufferGenerator& setReadoutMode(const FEDReadoutMode newReadoutMode);
      FEDBufferGenerator& setHeaderType(const FEDHeaderType newHeaderType);
      FEDBufferGenerator& setBufferFormat(const FEDBufferFormat newBufferFormat);
      FEDBufferGenerator& setDAQEventType(const FEDDAQEventType newDAQEventType);
      //disabled FE units produce no data at all
      //disabled channels have headers but data is all zeros (raw modes) or have no clusters (ZS)
      bool getFEUnitEnabled(const uint8_t internalFEUnitNumber) const;
      bool getChannelEnabled(const uint8_t internalFEDChannelNumber) const;
      FEDBufferGenerator& setFEUnitEnable(const uint8_t internalFEUnitNumber, const bool enabled);
      FEDBufferGenerator& setChannelEnable(const uint8_t internalFEDChannelNumber, const bool enabled);
      FEDBufferGenerator& setFEUnitEnables(const std::vector<bool>& feUnitsEnabled);
      FEDBufferGenerator& setChannelEnables(const std::vector<bool>& channelsEnabled);
      //make finer changes to defaults for parts of buffer
      //setting source ID in DAQ header and length and CRC in DAQ trailer has no effect since they are set when buffer is built
      FEDDAQHeader& daqHeader();
      FEDDAQTrailer& daqTrailer();
      TrackerSpecialHeader& trackerSpecialHeader();
      FEDFEHeader& feHeader();
      //method to generate buffer
      //unspecified parameters use defaults set by constructor or setters
      //FEDRawData object will be resized to fit buffer and filled
      void generateBuffer(FEDRawData* rawDataObject,
                          const FEDStripData& data,
                          uint16_t sourceID,
                          uint8_t packetCode) const;
    private:
      //method to fill buffer at pointer from pre generated components (only the length and CRC will be changed)
      //at least bufferSizeInBytes(feHeader,payload) must have already been allocated
      static void fillBuffer(uint8_t* pointerToStartOfBuffer,
                             const FEDDAQHeader& daqHeader,
                             const FEDDAQTrailer& daqTrailer,
                             const TrackerSpecialHeader& tkSpecialHeader,
                             const FEDFEHeader& feHeader,
                             const FEDBufferPayload& payload);
      //returns required size of buffer from given components
      static size_t bufferSizeInBytes(const FEDFEHeader& feHeader,
                                      const FEDBufferPayload& payload);
      //used to store default values
      FEDDAQHeader defaultDAQHeader_;
      FEDDAQTrailer defaultDAQTrailer_;
      TrackerSpecialHeader defaultTrackerSpecialHeader_;
      std::unique_ptr<FEDFEHeader> defaultFEHeader_;
      std::vector<bool> feUnitsEnabled_;
      std::vector<bool> channelsEnabled_;
    };
  
  //
  // Inline function definitions
  //
  
  //FEDStripData
  
  inline FEDStripData::FEDStripData(const std::vector<ChannelData>& data)
    : data_(data)
    { }
  
  //re-use non-const method
  inline FEDStripData::ChannelData& FEDStripData::channel(const uint8_t internalFEDChannelNum)
    {
      return const_cast<ChannelData&>(static_cast<const FEDStripData*>(this)->channel(internalFEDChannelNum));
    }
  
  inline FEDStripData::ChannelData& FEDStripData:: operator [] (const uint8_t internalFEDChannelNum)
    {
      return channel(internalFEDChannelNum);
    }
  
  inline const FEDStripData::ChannelData& FEDStripData:: operator [] (const uint8_t internalFEDChannelNum) const
    {
      return channel(internalFEDChannelNum);
    }
  
  inline FEDStripData::ChannelData::ChannelData(bool dataIsAlreadyConvertedTo8Bit, const size_t numberOfSamples,
                                                const std::pair<uint16_t,uint16_t> medians)
    : medians_(medians),
      data_(numberOfSamples,0),
      dataIs8Bit_(dataIsAlreadyConvertedTo8Bit)
    { }
  
  inline size_t FEDStripData::ChannelData::size() const
    {
      return data_.size();
    }
  
  inline const uint16_t& FEDStripData::ChannelData::operator [] (const size_t sampleNumber) const
    {
      return data_[sampleNumber];
    }
  
  //re-use const method
  inline uint16_t& FEDStripData::ChannelData::operator [] (const size_t sampleNumber)
    {
      return const_cast<uint16_t&>(static_cast<const ChannelData&>(*this)[sampleNumber]);
    }
  
  inline uint16_t FEDStripData::ChannelData::getSample(const uint16_t sampleNumber) const
  {
    //try {
    //  return data_.at(sampleNumber);
    //} catch (const std::out_of_range&) {
    //  std::ostringstream ss;
    //  ss << "Sample index out of range. "
    //     << "Requesting sample " << sampleNumber
    //     << " when channel has only " << data_.size() << " samples.";
    //  throw cms::Exception("FEDBufferGenerator") << ss.str();
    //}
    return data_[sampleNumber];
  }
  
  inline uint8_t FEDStripData::ChannelData::get8BitSample(const uint16_t sampleNumber, uint16_t nBotBitsToDrop) const
  {
    uint16_t sample = getSample(sampleNumber) >> nBotBitsToDrop;
    if (dataIs8Bit_) {
      return (0xFF & sample);
    }
    else {
      if (sample < 0xFE) return sample;
      else if (sample == 0x3FF) return 0xFF;
      else return 0xFE;
    }
  }
   
  inline uint16_t FEDStripData::ChannelData::get10BitSample(const uint16_t sampleNumber) const
  {
    if (dataIs8Bit_) {
      return (0xFF & getSample(sampleNumber));
    }
    else {
      const uint16_t sample = getSample(sampleNumber);
      if (sample < 0x3FF) return sample;
      else return 0x3FF;
    }
  }
  
  inline std::pair<uint16_t,uint16_t> FEDStripData::ChannelData::getMedians() const
    {
      return medians_;
    }
  
  inline void FEDStripData::ChannelData::setMedians(const std::pair<uint16_t,uint16_t> values)
    {
      medians_ = values;
    }
  
  //FEDBufferPayload
  
  inline size_t FEDBufferPayload::lengthInBytes() const
    {
      return data_.size();
    }
  
  inline void FEDBufferPayload::appendToBuffer(size_t* pIndexInBuffer, const uint8_t value)
    {
      data_[((*pIndexInBuffer)++)^7] = value;
    }
  
  inline void FEDBufferPayload::appendToBuffer(size_t* pIndexInBuffer, std::vector<uint8_t>::const_iterator start, std::vector<uint8_t>::const_iterator finish)
    {
      for (std::vector<uint8_t>::const_iterator iVal = start; iVal != finish; iVal++) {
        appendToBuffer(pIndexInBuffer,*iVal);
      }
    }
  
  //FEDBufferPayloadCreator
  
  inline FEDBufferPayloadCreator::FEDBufferPayloadCreator(const std::vector<bool>& feUnitsEnabled, const std::vector<bool>& channelsEnabled)
    : feUnitsEnabled_(feUnitsEnabled),
      channelsEnabled_(channelsEnabled)
    {}
  
  inline FEDBufferPayload FEDBufferPayloadCreator::operator () (FEDReadoutMode mode, uint8_t packetCode, const FEDStripData& data) const
    {
      return createPayload(mode, packetCode, data);
    }
  
  //FEDBufferGenerator
  
  inline uint32_t FEDBufferGenerator::getL1ID() const
    {
      return defaultDAQHeader_.l1ID();
    }
  
  inline uint16_t FEDBufferGenerator::getBXID() const
    {
      return defaultDAQHeader_.bxID();
    }
  
  inline FEDReadoutMode FEDBufferGenerator::getReadoutMode() const
    {
      return defaultTrackerSpecialHeader_.readoutMode();
    }
  
  inline FEDHeaderType FEDBufferGenerator::getHeaderType() const
    {
      return defaultTrackerSpecialHeader_.headerType();
    }
  
  inline FEDBufferFormat FEDBufferGenerator::getBufferFormat() const
    {
      return defaultTrackerSpecialHeader_.bufferFormat();
    }
  
  inline FEDDAQEventType FEDBufferGenerator::getDAQEventType() const
    {
      return defaultDAQHeader_.eventType();
    }
  
  inline FEDBufferGenerator& FEDBufferGenerator::setL1ID(const uint32_t newL1ID)
    {
      defaultDAQHeader_.setL1ID(newL1ID);
      return *this;
    }
  
  inline FEDBufferGenerator& FEDBufferGenerator::setBXID(const uint16_t newBXID)
    {
      defaultDAQHeader_.setBXID(newBXID);
      return *this;
    }
  
  inline FEDBufferGenerator& FEDBufferGenerator::setReadoutMode(const FEDReadoutMode newReadoutMode)
    {
      defaultTrackerSpecialHeader_.setReadoutMode(newReadoutMode);
      return *this;
    }
  
  inline FEDBufferGenerator& FEDBufferGenerator::setHeaderType(const FEDHeaderType newHeaderType)
    {
      defaultTrackerSpecialHeader_.setHeaderType(newHeaderType);
      return *this;
    }
  
  inline FEDBufferGenerator& FEDBufferGenerator::setBufferFormat(const FEDBufferFormat newBufferFormat)
    {
      defaultTrackerSpecialHeader_.setBufferFormat(newBufferFormat);
      return *this;
    }
  
  inline FEDBufferGenerator& FEDBufferGenerator::setDAQEventType(const FEDDAQEventType newDAQEventType)
    {
      defaultDAQHeader_.setEventType(newDAQEventType);
      return *this;
    }
  
  inline FEDDAQHeader& FEDBufferGenerator::daqHeader()
    {
      return defaultDAQHeader_;
    }
  
  inline FEDDAQTrailer& FEDBufferGenerator::daqTrailer()
    {
      return defaultDAQTrailer_;
    }
  
  inline TrackerSpecialHeader& FEDBufferGenerator::trackerSpecialHeader()
    {
      return defaultTrackerSpecialHeader_;
    }
  
  inline FEDFEHeader& FEDBufferGenerator::feHeader()
    {
      return *defaultFEHeader_;
    }
  
  inline size_t FEDBufferGenerator::bufferSizeInBytes(const FEDFEHeader& feHeader,
                                                             const FEDBufferPayload& payload)
    {
      //FE header + payload + tracker special header + daq header + daq trailer
      return feHeader.lengthInBytes()+payload.lengthInBytes()+8+8+8;
    }
  
}

#endif //ndef EventFilter_SiStripRawToDigi_FEDBufferGenerator_H
