#include "SiStripRawToClustersLazyUnpacker.h"


#include "SiStripRawToDigiUnpacker.h"

#include <sstream>
#include <iostream>


namespace sistrip { 
  
  RawToClustersLazyUnpacker::RawToClustersLazyUnpacker(const SiStripRegionCabling& regioncabling, StripClusterizerAlgorithm& clustalgo, SiStripRawProcessingAlgorithms& rpAlgos, const FEDRawDataCollection& data, bool dump) :

    raw_(&data),
    regions_(&(regioncabling.getRegionCabling())),
    clusterizer_(&clustalgo),
    rawAlgos_(&rpAlgos),
    buffers_(),
    dump_(dump),
    doAPVEmulatorCheck_(true)
  {
    buffers_.assign(1024,static_cast<sistrip::FEDBuffer*>(0));
  }

  RawToClustersLazyUnpacker::~RawToClustersLazyUnpacker() {

    std::vector< sistrip::FEDBuffer*>::iterator ibuffer = buffers_.begin();
    for (; ibuffer!=buffers_.end(); ibuffer++) {
      if (*ibuffer) delete *ibuffer;
    }
  }

  void RawToClustersLazyUnpacker::fill(const uint32_t& index, record_type& record) {

    // Get region, subdet and layer from element-index
    uint32_t region = SiStripRegionCabling::region(index);
    uint32_t subdet = static_cast<uint32_t>(SiStripRegionCabling::subdet(index));
    uint32_t layer = SiStripRegionCabling::layer(index);
 
    // Retrieve cabling for element
    const SiStripRegionCabling::ElementCabling& element = (*regions_)[region][subdet][layer];
  
    // Loop dets
    SiStripRegionCabling::ElementCabling::const_iterator idet = element.begin();
    for (;idet!=element.end();idet++) {
    
      // If det id is null or invalid continue.
      if ( ( (!(idet->first)) | (idet->first == sistrip::invalid32_)) || !clusterizer_->stripByStripBegin(idet->first)) { continue; }
    
      // Loop over apv-pairs of det
      std::vector<FedChannelConnection>::const_iterator iconn = idet->second.begin();
      for (;iconn!=idet->second.end();iconn++) {
        const uint16_t fedId = iconn->fedId();
      
	// If fed id is null or connection is invalid continue
	if ( !fedId || !iconn->isConnected() ) { continue; }    
      
	// If Fed hasnt already been initialised, extract data and initialise
        FEDBuffer* buffer = buffers_[fedId];
	if (!buffer) {
	
	  // Retrieve FED raw data for given FED
          const FEDRawData& rawData = raw_->FEDData( static_cast<int>(fedId) );
	
	  // Check on FEDRawData pointer
	  if ( !rawData.data() ) {
            if (edm::isDebugEnabled()) {
              edm::LogWarning(sistrip::mlRawToCluster_)
                << "[sistrip::RawToClustersLazyGetter::" 
                << __func__ 
                << "]"
                << " NULL pointer to FEDRawData for FED id " 
                << fedId;
            }
	    continue;
	  }	
	
	  // Check on FEDRawData size
	  if ( !rawData.size() ) {
	    if (edm::isDebugEnabled()) {
              edm::LogWarning(sistrip::mlRawToCluster_)
                << "[sistrip::RawToClustersLazyGetter::" 
                << __func__ << "]"
                << " FEDRawData has zero size for FED id " 
                << fedId;
            }
	    continue;
	  }
	
	  // construct FEDBuffer
	  try {
            buffers_[fedId] = buffer = new sistrip::FEDBuffer(rawData.data(),rawData.size());
            if (!buffer->doChecks(false)) throw cms::Exception("FEDBuffer") << "FED Buffer check fails for FED ID" << fedId << ".";
          }
	  catch (const cms::Exception& e) { 
            if (edm::isDebugEnabled()) {
              edm::LogWarning(sistrip::mlRawToCluster_) 
                << "Exception caught when creating FEDBuffer object for FED " << fedId << ": " << e.what();
            }
	    if ( buffer ) { delete buffer; }
	    buffers_[fedId] = 0;
	    continue;
	  }

	  // dump of FEDRawData to stdout
	  if ( dump_ ) {
	    std::stringstream ss;
	    RawToDigiUnpacker::dumpRawData( fedId, rawData, ss );
	    LogTrace(mlRawToDigi_) 
	      << ss.str();
	  }
        }

	// check channel
        const uint8_t fedCh = iconn->fedCh();

	if (!buffer->channelGood(fedCh,doAPVEmulatorCheck_)) {
          if (edm::isDebugEnabled()) {
            std::ostringstream ss;
            ss << "Problem unpacking channel " << fedCh << " on FED " << fedId;
            edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
          }
          continue;
        }
      
	// Determine APV std::pair number
	uint16_t ipair = iconn->apvPairNumber();

        const sistrip::FEDReadoutMode mode = buffer->readoutMode();
	if (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED ) { 
	
          try {
	    // create unpacker
	    sistrip::FEDZSChannelUnpacker unpacker = sistrip::FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(buffer->channel(fedCh));
	    
	    // unpack
	    clusterizer_->addFed(unpacker,ipair,record);
	    /*
	    while (unpacker.hasData()) {
	      clusterizer_->stripByStripAdd(unpacker.sampleNumber()+ipair*256,unpacker.adc(),record);
	      unpacker++;
	    }
            */
          } catch (const cms::Exception& e) {
            if (edm::isDebugEnabled()) {
              std::ostringstream ss;
              ss << "Unordered clusters for channel " << fedCh << " on FED " << fedId << ": " << e.what();
              edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
            }
            continue;
          }
	}

	else if (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE ) { 
	
          try {
            // create unpacker
            sistrip::FEDZSChannelUnpacker unpacker = sistrip::FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(buffer->channel(fedCh));

            // unpack
	    clusterizer_->addFed(unpacker,ipair,record);
	    /*
            while (unpacker.hasData()) {
              clusterizer_->stripByStripAdd(unpacker.sampleNumber()+ipair*256,unpacker.adc(),record);
              unpacker++;
            }
	    */
          } catch (const cms::Exception& e) {
            if (edm::isDebugEnabled()) {
              std::ostringstream ss;
              ss << "Unordered clusters for channel " << fedCh << " on FED " << fedId << ": " << e.what();
              edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
            }                                               
            continue;
          }
	}

	else if (mode == sistrip::READOUT_MODE_VIRGIN_RAW ) {

	  // create unpacker
	  sistrip::FEDRawChannelUnpacker unpacker = sistrip::FEDRawChannelUnpacker::virginRawModeUnpacker(buffer->channel(fedCh));

	  // unpack
	  std::vector<int16_t> digis;
	  while (unpacker.hasData()) {
	    digis.push_back(unpacker.adc());
	    unpacker++;
	  }

	  //process raw
	  uint32_t id = iconn->detId();
          edm::DetSet<SiStripDigi> zsdigis(id);
	  //rawAlgos_->subtractorPed->subtract( id, ipair*256, digis);
	  //rawAlgos_->subtractorCMN->subtract( id, digis);
	  //rawAlgos_->suppressor->suppress( digis, zsdigis);
	  uint16_t firstAPV = ipair*2;
	  rawAlgos_->SuppressVirginRawData(id, firstAPV,digis, zsdigis);  
         for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
	    clusterizer_->stripByStripAdd( it->strip(), it->adc(), record);
	  }
	}

	else if (mode == sistrip::READOUT_MODE_PROC_RAW ) {

	  // create unpacker
	  sistrip::FEDRawChannelUnpacker unpacker = sistrip::FEDRawChannelUnpacker::procRawModeUnpacker(buffer->channel(fedCh));

	  // unpack
	  std::vector<int16_t> digis;
	  while (unpacker.hasData()) {
	    digis.push_back(unpacker.adc());
	    unpacker++;
	  }

	  //process raw
	  uint32_t id = iconn->detId();
          edm::DetSet<SiStripDigi> zsdigis(id);
	  //rawAlgos_->subtractorCMN->subtract( id, digis);
	  //rawAlgos_->suppressor->suppress( digis, zsdigis);
           uint16_t firstAPV = ipair*2;
          rawAlgos_->SuppressProcessedRawData(id, firstAPV,digis, zsdigis); 
	  for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
	    clusterizer_->stripByStripAdd( it->strip(), it->adc(), record);
	  }
	}

	else {
	  edm::LogWarning(sistrip::mlRawToCluster_)
	    << "[sistrip::RawToClustersLazyGetter::" 
	    << __func__ << "]"
	    << " FEDRawData readout mode "
	    << mode
	    << " from FED id "
	    << fedId 
	    << " not supported."; 
	  continue;
	}
      }
      clusterizer_->stripByStripEnd(record);
    }
  }

  
}
