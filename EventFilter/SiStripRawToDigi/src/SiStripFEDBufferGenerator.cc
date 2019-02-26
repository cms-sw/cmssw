#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace sistrip {
  
  //FEDStripData
  
  FEDStripData::FEDStripData(bool dataIsAlreadyConvertedTo8Bit, const size_t samplesPerChannel)
    : data_(FEDCH_PER_FED,ChannelData(dataIsAlreadyConvertedTo8Bit,samplesPerChannel))
  {
    if (samplesPerChannel > SCOPE_MODE_MAX_SCOPE_LENGTH) {
      std::ostringstream ss;
      ss << "Scope length " << samplesPerChannel << " is too long. "
         << "Max scope length is " << SCOPE_MODE_MAX_SCOPE_LENGTH << ".";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }
  
  const FEDStripData::ChannelData& FEDStripData::channel(const uint8_t internalFEDChannelNum) const
  {
    try {
      return data_.at(internalFEDChannelNum);
    } catch (const std::out_of_range&) {
      std::ostringstream ss;
      ss << "Channel index out of range. (" << uint16_t(internalFEDChannelNum) << ") "
         << "Index should be in internal numbering scheme (0-95). ";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }
  
  void FEDStripData::ChannelData::setSample(const uint16_t sampleNumber, const uint16_t value)
  {
    if (value > 0x3FF) {
      std::ostringstream ss;
      ss << "Sample value (" << value << ") is too large. Maximum allowed is 1023. ";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
    try {
      data_.at(sampleNumber) = value;
    } catch (const std::out_of_range&) {
      std::ostringstream ss;
      ss << "Sample index out of range. "
         << "Requesting sample " << sampleNumber
         << " when channel has only " << data_.size() << " samples.";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }
  
  //FEDBufferPayload
  
  FEDBufferPayload::FEDBufferPayload(const std::vector< std::vector<uint8_t> >& channelBuffers)
  {
    //calculate size of buffer and allocate enough memory
    uint32_t totalSize = 0;
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      for (uint8_t iCh = 0; iCh < FEDCH_PER_FEUNIT; iCh++) {
        totalSize += channelBuffers[iFE*FEDCH_PER_FEUNIT+iCh].size();
      }
      //if it does not finish on a 64Bit word boundary then take into account padding
      if (totalSize%8) {
        totalSize = ((totalSize/8) + 1)*8;
      }
    }
    data_.resize(totalSize);
    size_t indexInBuffer = 0;
    feLengths_.reserve(FEUNITS_PER_FED);
    //copy channel data into buffer with padding and update lengths
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      const size_t lengthAtStartOfFEUnit = indexInBuffer;
      //insert data for FE unit
      for (uint8_t iCh = 0; iCh < FEDCH_PER_FEUNIT; iCh++) {
        appendToBuffer(&indexInBuffer,channelBuffers[iFE*FEDCH_PER_FEUNIT+iCh].begin(),channelBuffers[iFE*FEDCH_PER_FEUNIT+iCh].end());
      }
      //store length
      feLengths_.push_back(indexInBuffer-lengthAtStartOfFEUnit);
      //add padding
      while (indexInBuffer % 8) appendToBuffer(&indexInBuffer,0);
    }
  }
  
  const uint8_t* FEDBufferPayload::data() const
  {
    //vectors are guarenteed to be contiguous
    if (lengthInBytes()) return &data_[0];
    //return NULL if there is no data yet
    else return nullptr;
  }
  
  uint16_t FEDBufferPayload::getFELength(const uint8_t internalFEUnitNum) const
  {
    try{
      return feLengths_.at(internalFEUnitNum);
    } catch (const std::out_of_range&) {
      std::ostringstream ss;
      ss << "Invalid FE unit number " << internalFEUnitNum << ". "
         << "Number should be in internal numbering scheme (0-7). ";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }

  FEDBufferPayload FEDBufferPayloadCreator::createPayload(FEDReadoutMode mode, uint8_t packetCode, const FEDStripData& data) const
  {
    std::vector< std::vector<uint8_t> > channelBuffers(FEDCH_PER_FED,std::vector<uint8_t>());
    for (size_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
      if (!feUnitsEnabled_[iCh/FEDCH_PER_FEUNIT]) continue;
      fillChannelBuffer(&channelBuffers[iCh], mode, packetCode, data.channel(iCh), channelsEnabled_[iCh]);
    }
    return FEDBufferPayload(channelBuffers);
  }

  void FEDBufferPayloadCreator::fillChannelBuffer(std::vector<uint8_t>* channelBuffer, FEDReadoutMode mode, uint8_t packetCode, const FEDStripData::ChannelData& data, const bool channelEnabled) const
  {
    switch (mode) {
    case READOUT_MODE_SCOPE:
      fillRawChannelBuffer(channelBuffer,PACKET_CODE_SCOPE,data,channelEnabled,false);
      break;
    case READOUT_MODE_VIRGIN_RAW:
      switch (packetCode) {
        case PACKET_CODE_VIRGIN_RAW:
          fillRawChannelBuffer(channelBuffer,PACKET_CODE_VIRGIN_RAW,data,channelEnabled,true);
          break;
        case PACKET_CODE_VIRGIN_RAW10:
          fillRawChannelBuffer(channelBuffer,PACKET_CODE_VIRGIN_RAW10,data,channelEnabled,true);
          break;
        case PACKET_CODE_VIRGIN_RAW8_BOTBOT:
          fillRawChannelBuffer(channelBuffer,PACKET_CODE_VIRGIN_RAW8_BOTBOT,data,channelEnabled,true);
          break;
        case PACKET_CODE_VIRGIN_RAW8_TOPBOT:
          fillRawChannelBuffer(channelBuffer,PACKET_CODE_VIRGIN_RAW8_TOPBOT,data,channelEnabled,true);
        break;
        }
      break;
    case READOUT_MODE_PROC_RAW:
      fillRawChannelBuffer(channelBuffer,PACKET_CODE_PROC_RAW,data,channelEnabled,false);
      break;
    case READOUT_MODE_ZERO_SUPPRESSED:
    //case READOUT_MODE_ZERO_SUPPRESSED_CMOVERRIDE:
      fillZeroSuppressedChannelBuffer(channelBuffer,packetCode,data,channelEnabled);
      break;
    case READOUT_MODE_ZERO_SUPPRESSED_LITE10:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT:
    case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE:
      fillZeroSuppressedLiteChannelBuffer(channelBuffer,data,channelEnabled,mode);
      break;
    case READOUT_MODE_PREMIX_RAW:
      fillPreMixRawChannelBuffer(channelBuffer,data,channelEnabled);
      break;
    default:
      std::ostringstream ss;
      ss << "Invalid readout mode " << mode;
      throw cms::Exception("FEDBufferGenerator") << ss.str();
      break;
    }
  }

  void FEDBufferPayloadCreator::fillRawChannelBuffer(std::vector<uint8_t>* channelBuffer,
                                                    const uint8_t packetCode,
                                                    const FEDStripData::ChannelData& data,
                                                    const bool channelEnabled,
                                                    const bool reorderData) const
  {
    const uint16_t nSamples = data.size();
    uint16_t channelLength = 0;
    switch (packetCode) {
      case PACKET_CODE_VIRGIN_RAW:
        channelLength = nSamples*2 + 3;
        break;
      case PACKET_CODE_VIRGIN_RAW10:
        channelLength = std::ceil(nSamples*1.25) + 3;
        break;
      case PACKET_CODE_VIRGIN_RAW8_BOTBOT:
      case PACKET_CODE_VIRGIN_RAW8_TOPBOT:
        channelLength = nSamples*1 + 3;
        break;
    }
    channelBuffer->reserve(channelLength);
    //length (max length is 0xFFF)
    channelBuffer->push_back( channelLength & 0xFF );
    channelBuffer->push_back( (channelLength & 0xF00) >> 8 );
    //packet code
    channelBuffer->push_back(packetCode);
    //channel samples
    uint16_t sampleValue_pre = 0;
    for (uint16_t sampleNumber = 0; sampleNumber < nSamples; sampleNumber++) {
      const uint16_t sampleIndex = ( reorderData ? FEDStripOrdering::physicalOrderForStripInChannel(sampleNumber) : sampleNumber );
      const uint16_t sampleValue = (channelEnabled ? data.getSample(sampleIndex) : 0);
      switch (packetCode) {
        case PACKET_CODE_VIRGIN_RAW:
          channelBuffer->push_back(sampleValue & 0xFF);
          channelBuffer->push_back((sampleValue & 0x300) >> 8);
          break;
        case PACKET_CODE_VIRGIN_RAW10:
          if (sampleNumber%4==0) {
            channelBuffer->push_back((sampleValue & 0x3FC) >> 2);
          }
          else if (sampleNumber%4==1) {
            channelBuffer->push_back(((sampleValue_pre & 0x3) << 6) | ((sampleValue & 0x3F0) >> 4));
          }
          else if (sampleNumber%4==2) {
            channelBuffer->push_back(((sampleValue_pre & 0xF) << 4) | ((sampleValue & 0x3C0) >> 6));
          }
          else if (sampleNumber%4==3) {
            channelBuffer->push_back(((sampleValue_pre & 0x3F) << 2) | ((sampleValue & 0x300)>>8));
            channelBuffer->push_back(sampleValue & 0xFF);
          }
          sampleValue_pre = sampleValue;
          break;
        case PACKET_CODE_VIRGIN_RAW8_BOTBOT:
          channelBuffer->push_back((sampleValue & 0x3FC) >> 2);
          break;
        case PACKET_CODE_VIRGIN_RAW8_TOPBOT:
          channelBuffer->push_back((sampleValue & 0x1FE) >> 1);
          break;
      }
    }
  }

  void FEDBufferPayloadCreator::fillZeroSuppressedChannelBuffer(std::vector<uint8_t>* channelBuffer,
                                                               const uint8_t packetCode,
                                                               const FEDStripData::ChannelData& data,
                                                               const bool channelEnabled) const
  {
    channelBuffer->reserve(50);
    //if channel is disabled then create empty channel header and return
    if (!channelEnabled) {
      //min length 7
      channelBuffer->push_back(7);
      channelBuffer->push_back(0);
      //packet code
      channelBuffer->push_back(packetCode);
      //4 bytes of medians
      channelBuffer->insert(channelBuffer->end(),4,0);
      return;
    }
    //if channel is not empty
    //add space for channel length
    channelBuffer->push_back(0xFF);
    channelBuffer->push_back(0xFF);
    //packet code
    channelBuffer->push_back(packetCode);
    //add medians
    const std::pair<uint16_t,uint16_t> medians = data.getMedians();
    channelBuffer->push_back(medians.first & 0xFF);
    channelBuffer->push_back((medians.first & 0x300) >> 8);
    channelBuffer->push_back(medians.second & 0xFF);
    channelBuffer->push_back((medians.second & 0x300) >> 8);
    //clusters
    fillClusterData(channelBuffer, packetCode, data, READOUT_MODE_ZERO_SUPPRESSED);
    //set length
    const uint16_t length = channelBuffer->size();
    (*channelBuffer)[0] = (length & 0xFF);
    (*channelBuffer)[1] = ((length & 0x300) >> 8);
  }

  void FEDBufferPayloadCreator::fillZeroSuppressedLiteChannelBuffer(std::vector<uint8_t>* channelBuffer,
                                                                   const FEDStripData::ChannelData& data,
                                                                   const bool channelEnabled,
                                                                   const FEDReadoutMode mode) const
  {
    channelBuffer->reserve(50);
    //if channel is disabled then create empty channel header and return
    if (!channelEnabled) {
      //min length 2
      channelBuffer->push_back(2);
      channelBuffer->push_back(0);
      return;
    }
    //if channel is not empty
    //add space for channel length
    channelBuffer->push_back(0xFF);
    channelBuffer->push_back(0xFF);
    //clusters
    fillClusterData(channelBuffer, 0, data, mode);
    //set fibre length
    const uint16_t length = channelBuffer->size();
    (*channelBuffer)[0] = (length & 0xFF);
    (*channelBuffer)[1] = ((length & 0x300) >> 8);
  }

  void FEDBufferPayloadCreator::fillPreMixRawChannelBuffer(std::vector<uint8_t>* channelBuffer,
                                                                   const FEDStripData::ChannelData& data,
                                                                   const bool channelEnabled) const
  {
   channelBuffer->reserve(50);
    //if channel is disabled then create empty channel header and return
    if (!channelEnabled) {
      //min length 7
      channelBuffer->push_back(7);
      channelBuffer->push_back(0);
      //packet code
      channelBuffer->push_back(PACKET_CODE_ZERO_SUPPRESSED);
      //4 bytes of medians
      channelBuffer->insert(channelBuffer->end(),4,0);
      return;
    }
    //if channel is not empty
    //add space for channel length
    channelBuffer->push_back(0xFF); channelBuffer->push_back(0xFF);
    //packet code
    channelBuffer->push_back(PACKET_CODE_ZERO_SUPPRESSED);
    //add medians
    const std::pair<uint16_t,uint16_t> medians = data.getMedians();
    channelBuffer->push_back(medians.first & 0xFF);
    channelBuffer->push_back((medians.first & 0x300) >> 8);
    channelBuffer->push_back(medians.second & 0xFF);
    channelBuffer->push_back((medians.second & 0x300) >> 8);
    //clusters
    fillClusterDataPreMixMode(channelBuffer,data);
    //set length
    const uint16_t length = channelBuffer->size();
    (*channelBuffer)[0] = (length & 0xFF);
    (*channelBuffer)[1] = ((length & 0x300) >> 8);
  }

  void FEDBufferPayloadCreator::fillClusterData(std::vector<uint8_t>* channelBuffer, uint8_t packetCode, const FEDStripData::ChannelData& data, const FEDReadoutMode mode) const
  {
    // ZS lite: retrieve "packet code"
    switch (mode) {
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8:
        packetCode = PACKET_CODE_ZERO_SUPPRESSED;
        break;
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE:
        packetCode = PACKET_CODE_ZERO_SUPPRESSED8_TOPBOT;
        break;
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE:
        packetCode = PACKET_CODE_ZERO_SUPPRESSED8_BOTBOT;
        break;
      case READOUT_MODE_ZERO_SUPPRESSED_LITE10:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE:
        packetCode = PACKET_CODE_ZERO_SUPPRESSED10;
        break;
      default: ;
    }
    const bool is10Bit = ( packetCode == PACKET_CODE_ZERO_SUPPRESSED10 );
    const uint16_t bShift = ( packetCode == PACKET_CODE_ZERO_SUPPRESSED8_BOTBOT ? 2
                          : ( packetCode == PACKET_CODE_ZERO_SUPPRESSED8_TOPBOT ? 1 : 0 ) );

    uint16_t clusterSize = 0; // counter
    std::size_t size_pos = 0; // index of cluster size
    uint16_t adc_pre = 0;
    const uint16_t nSamples = data.size();
    for( uint16_t strip = 0; strip < nSamples; ++strip) {
      const uint16_t adc = is10Bit ? data.get10BitSample(strip) : data.get8BitSample(strip, bShift);
      if (adc) {
	if ( clusterSize==0 || strip == STRIPS_PER_APV ) {
	  if (clusterSize) {
            if ( is10Bit && (clusterSize%4) ) { channelBuffer->push_back(adc_pre); }
            (*channelBuffer)[size_pos] = clusterSize;
	    clusterSize = 0;
	  }
          // cluster header: first strip and size
	  channelBuffer->push_back(strip);
          size_pos = channelBuffer->size();
	  channelBuffer->push_back(0); // for clustersize
	}
        if ( ! is10Bit ) {
          channelBuffer->push_back(adc & 0xFF);
        } else {
          if (clusterSize%4==0) {
            channelBuffer->push_back((adc & 0x3FC) >> 2);
            adc_pre = ((adc & 0x3) << 6);
          } else if (clusterSize%4==1) {
            channelBuffer->push_back(adc_pre | ((adc & 0x3F0) >> 4));
            adc_pre = ((adc & 0xF) << 4);
          } else if (clusterSize%4==2) {
            channelBuffer->push_back(adc_pre | ((adc & 0x3C0) >> 6));
            adc_pre = ((adc & 0x3F) << 2);
          } else if (clusterSize%4==3) {
            channelBuffer->push_back(adc_pre | ((adc & 0x300) >> 8));
            channelBuffer->push_back(adc & 0xFF);
            adc_pre = 0;
          }
        }
	++clusterSize;
      }
      else if (clusterSize) {
        if ( is10Bit && (clusterSize%4) ) { channelBuffer->push_back(adc_pre); }
        (*channelBuffer)[size_pos] = clusterSize;
        clusterSize = 0;
      }
    }
    if(clusterSize) {
      (*channelBuffer)[size_pos] = clusterSize;
      if ( is10Bit && (clusterSize%4) ) { channelBuffer->push_back(adc_pre); }
    }
  }

  void FEDBufferPayloadCreator::fillClusterDataPreMixMode(std::vector<uint8_t>* channelBuffer, const FEDStripData::ChannelData& data) const
  {
    uint16_t clusterSize = 0;
    const uint16_t nSamples = data.size();
    for( uint16_t strip = 0; strip < nSamples; ++strip) {
      const uint16_t adc = data.get10BitSample(strip);

      if(adc) {
	if( clusterSize==0 || strip == STRIPS_PER_APV ) { 
	  if(clusterSize) { 
	    *(channelBuffer->end() - 2*clusterSize - 1) = clusterSize ; 
	    clusterSize = 0; 
	  }
	  channelBuffer->push_back(strip); 
	  channelBuffer->push_back(0); //clustersize	  
	}
	channelBuffer->push_back(adc & 0xFF);
	channelBuffer->push_back((adc & 0x0300) >> 8);

	++clusterSize;
      }

      else if(clusterSize) { 
	*(channelBuffer->end() - 2*clusterSize - 1) = clusterSize ; 
	clusterSize = 0; 
      }
    }
    if(clusterSize) {
      *(channelBuffer->end() - 2*clusterSize - 1) = clusterSize ;
    }
  }

  //FEDBufferGenerator
  
  FEDBufferGenerator::FEDBufferGenerator(const uint32_t l1ID, const uint16_t bxID,
                                         const std::vector<bool>& feUnitsEnabled, const std::vector<bool>& channelsEnabled,
                                         const FEDReadoutMode readoutMode, const FEDHeaderType headerType, const FEDBufferFormat bufferFormat,
                                         const FEDDAQEventType evtType)
    : defaultDAQHeader_(l1ID,bxID,0,evtType),
      defaultDAQTrailer_(0,0),
      defaultTrackerSpecialHeader_(bufferFormat,readoutMode,headerType),
      defaultFEHeader_(FEDFEHeader::newFEHeader(headerType)),
      feUnitsEnabled_(feUnitsEnabled),
      channelsEnabled_(channelsEnabled)
  {
    if (!defaultFEHeader_.get()) {
      std::ostringstream ss;
      ss << "Bad header format: " << headerType;
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }
  
  bool FEDBufferGenerator::getFEUnitEnabled(const uint8_t internalFEUnitNumber) const
  {
    try {
      return feUnitsEnabled_.at(internalFEUnitNumber);
    } catch (const std::out_of_range&) {
      std::ostringstream ss;
      ss << "Invalid FE unit number " << internalFEUnitNumber << ". Should be in internal numbering scheme (0-7)";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }
  
  bool FEDBufferGenerator::getChannelEnabled(const uint8_t internalFEDChannelNumber) const
  {
    try {
      return channelsEnabled_.at(internalFEDChannelNumber);
    } catch (const std::out_of_range&) {
      
      std::ostringstream ss;
      ss << "Invalid channel number " << internalFEDChannelNumber << ". "
         << "Should be in internal numbering scheme (0-95)";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
  }
  
  FEDBufferGenerator& FEDBufferGenerator::setFEUnitEnable(const uint8_t internalFEUnitNumber, const bool enabled)
  {
    try {
      feUnitsEnabled_.at(internalFEUnitNumber) = enabled;
    } catch (const std::out_of_range&) {
      std::ostringstream ss;
      ss << "Invalid FE unit number " << internalFEUnitNumber << ". "
         << "Should be in internal numbering scheme (0-7)";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
    return *this;
  }
  
  FEDBufferGenerator& FEDBufferGenerator::setChannelEnable(const uint8_t internalFEDChannelNumber, const bool enabled)
  {
    try {
      channelsEnabled_.at(internalFEDChannelNumber) = enabled;
    } catch (const std::out_of_range&) {
      std::ostringstream ss;
      ss << "Invalid channel number " << internalFEDChannelNumber << ". "
         <<"Should be in internal numbering scheme (0-95)";
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
    return *this;
  }
  
  FEDBufferGenerator& FEDBufferGenerator::setFEUnitEnables(const std::vector<bool>& feUnitEnables)
  {
    if (feUnitEnables.size() != FEUNITS_PER_FED) {
      std::ostringstream ss;
      ss << "Setting FE enable vector with vector which is the wrong size. Size is " << feUnitEnables.size()
         << " it must be " << FEUNITS_PER_FED << "." << std::endl;
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
    feUnitsEnabled_ = feUnitEnables;
    return *this;
  }
  
  FEDBufferGenerator& FEDBufferGenerator::setChannelEnables(const std::vector<bool>& channelEnables)
  {
    if (channelEnables.size() != FEDCH_PER_FED) {
      std::ostringstream ss;
      ss << "Setting FED channel enable vector with vector which is the wrong size. Size is " << channelEnables.size()
         << " it must be " << FEDCH_PER_FED << "." << std::endl;
      throw cms::Exception("FEDBufferGenerator") << ss.str();
    }
    channelsEnabled_ = channelEnables;
    return *this;
  }
  
  void FEDBufferGenerator::generateBuffer(FEDRawData* rawDataObject, const FEDStripData& data, uint16_t sourceID, uint8_t packetCode) const
  {
    //deal with disabled FE units and channels properly (FE enables, status bits)
    TrackerSpecialHeader tkSpecialHeader(defaultTrackerSpecialHeader_);
    std::unique_ptr<FEDFEHeader> fedFeHeader(defaultFEHeader_->clone());
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      const bool enabled = feUnitsEnabled_[iFE];
      tkSpecialHeader.setFEEnableForFEUnit(iFE,enabled);
      if (!enabled) {
        for (uint8_t iFEUnitChannel = 0; iFEUnitChannel < FEDCH_PER_FEUNIT; iFEUnitChannel++) {
          fedFeHeader->setChannelStatus(iFE,iFEUnitChannel,FEDChannelStatus(0));
        }
      }
    }
    for (uint8_t iCh = 0; iCh < FEDCH_PER_FED; iCh++) {
      if (!channelsEnabled_[iCh]) {
        fedFeHeader->setChannelStatus(iCh,FEDChannelStatus(0));
      }
    }
    //set the source ID
    FEDDAQHeader daqHeader(defaultDAQHeader_);
    daqHeader.setSourceID(sourceID);
    //build payload
    const FEDBufferPayloadCreator payloadPacker(feUnitsEnabled_,channelsEnabled_);
    const FEDBufferPayload payload = payloadPacker(getReadoutMode(), packetCode, data);
    //fill FE lengths
    for (uint8_t iFE = 0; iFE < FEUNITS_PER_FED; iFE++) {
      fedFeHeader->setFEUnitLength(iFE,payload.getFELength(iFE));
    }
    //resize buffer
    rawDataObject->resize(bufferSizeInBytes(*fedFeHeader,payload));
    //fill buffer
    fillBuffer(rawDataObject->data(), daqHeader, defaultDAQTrailer_, tkSpecialHeader, *fedFeHeader, payload);
  }
  
  void FEDBufferGenerator::fillBuffer(uint8_t* pointerToStartOfBuffer,
                                      const FEDDAQHeader& daqHeader,
                                      const FEDDAQTrailer& daqTrailer,
                                      const TrackerSpecialHeader& tkSpecialHeader,
                                      const FEDFEHeader& feHeader,
                                      const FEDBufferPayload& payload)
  {
    //set the length in the DAQ trailer
    const size_t lengthInBytes = bufferSizeInBytes(feHeader,payload);
    FEDDAQTrailer updatedDAQTrailer(daqTrailer);
    updatedDAQTrailer.setEventLengthIn64BitWords(lengthInBytes/8);
    //copy pieces into buffer in order
    uint8_t* bufferPointer = pointerToStartOfBuffer;
    memcpy(bufferPointer,daqHeader.data(),8);
    bufferPointer += 8;
    memcpy(bufferPointer,tkSpecialHeader.data(),8);
    bufferPointer += 8;
    memcpy(bufferPointer,feHeader.data(),feHeader.lengthInBytes());
    bufferPointer += feHeader.lengthInBytes();
    memcpy(bufferPointer,payload.data(),payload.lengthInBytes());
    bufferPointer += payload.lengthInBytes();
    memcpy(bufferPointer,updatedDAQTrailer.data(),8);
    //update CRC
    const uint16_t crc = calculateFEDBufferCRC(pointerToStartOfBuffer,lengthInBytes);
    updatedDAQTrailer.setCRC(crc);
    memcpy(bufferPointer,updatedDAQTrailer.data(),8);
    //word swap if necessary
    if (tkSpecialHeader.wasSwapped()) {
      for (size_t i = 0; i < 8; i++) {
        bufferPointer[i] = bufferPointer[i^4];
      }
    }
  }
  
}
