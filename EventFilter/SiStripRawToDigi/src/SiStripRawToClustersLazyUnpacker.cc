#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersLazyUnpacker.h"
#include <sstream>
#include <iostream>

OldSiStripRawToClustersLazyUnpacker::OldSiStripRawToClustersLazyUnpacker(const SiStripRegionCabling& regioncabling, const SiStripClusterizerFactory& clustfact, const FEDRawDataCollection& data) :

  raw_(&data),
  regions_(&(regioncabling.getRegionCabling())),
  clusterizer_(&clustfact),
  fedEvents_(),
  rawToDigi_(0,0,0,0,0),
  fedRawData_()
{
  fedEvents_.assign(1024,static_cast<Fed9U::Fed9UEvent*>(0));
}

OldSiStripRawToClustersLazyUnpacker::~OldSiStripRawToClustersLazyUnpacker() {
  std::vector< Fed9U::Fed9UEvent*>::iterator ifedevent = fedEvents_.begin();
  for (; ifedevent!=fedEvents_.end(); ifedevent++) {
    if (*ifedevent) {
      delete (*ifedevent);
      *ifedevent = 0;
    }
  }
}

void OldSiStripRawToClustersLazyUnpacker::fill(const uint32_t& index, record_type& record) {

  //Get region, subdet and layer from element-index
  uint32_t region = SiStripRegionCabling::region(index);
  uint32_t subdet = static_cast<uint32_t>(SiStripRegionCabling::subdet(index));
  uint32_t layer = SiStripRegionCabling::layer(index);
 
  //Retrieve cabling for element
  const SiStripRegionCabling::ElementCabling& element = (*regions_)[region][subdet][layer];
  
  //Loop dets
  SiStripRegionCabling::ElementCabling::const_iterator idet = element.begin();
  for (;idet!=element.end();idet++) {
    
    //If det id is null or invalid continue.
    if ( !(idet->first) || (idet->first == sistrip::invalid32_) ) { continue; }
    
    //Loop over apv-pairs of det
    std::vector<FedChannelConnection>::const_iterator iconn = idet->second.begin();
    for (;iconn!=idet->second.end();iconn++) {
      
      //If fed id is null or connection is invalid continue
      if ( !iconn->fedId() || !iconn->isConnected() ) { continue; }    
      
      //If Fed hasnt already been initialised, extract data and initialise
      if (!fedEvents_[iconn->fedId()]) {
	
	// Retrieve FED raw data for given FED
	const FEDRawData& input = raw_->FEDData( static_cast<int>(iconn->fedId()) );
	
	// Cache new correctly-ordered FEDRawData object (to maintain scope for Fed9UEvent)
	fedRawData_.push_back( FEDRawData() );
	rawToDigi_.unpacker()->locateStartOfFedBuffer( iconn->fedId(), input, fedRawData_.back() );

	// Recast data to suit Fed9UEvent
	Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( fedRawData_.back().data() ) );
	Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( fedRawData_.back().size() / 4 ); 
	
	// Check on FEDRawData pointer
	if ( !data_u32 ) {
	  if ( edm::isDebugEnabled() ) {
	    edm::LogWarning(sistrip::mlRawToCluster_)
	      << "[OldSiStripRawToClustersLazyGetter::" 
	      << __func__ 
	      << "]"
	      << " NULL pointer to FEDRawData for FED id " 
	      << iconn->fedId();
	  }
	  continue;
	}	
	
	// Check on FEDRawData size
	if ( !size_u32 ) {
	  if ( edm::isDebugEnabled() ) {
	    edm::LogWarning(sistrip::mlRawToCluster_)
	      << "[OldSiStripRawToClustersLazyGetter::" 
	      << __func__ << "]"
	      << " FEDRawData has zero size for FED id " 
	      << iconn->fedId();
	  }
	  continue;
	}
	
	// Construct Fed9UEvent using present FED buffer
	try {fedEvents_[iconn->fedId()] = new Fed9U::Fed9UEvent(data_u32,0,size_u32);} 
	catch(...) { 
	  rawToDigi_.unpacker()->handleException( __func__, "Problem when constructing Fed9UEvent" ); 
	  if ( fedEvents_[iconn->fedId()] ) { delete fedEvents_[iconn->fedId()]; }
	  fedEvents_[iconn->fedId()] = 0;
	  continue;
	}
      }
      
      //Calculate corresponding FED unit, channel
      uint16_t iunit = 0;
      uint16_t ichan = 0;
      uint16_t chan = 0;
      try {
	Fed9U::Fed9UAddress addr;
	addr.setFedChannel( static_cast<unsigned char>( iconn->fedCh() ) );
        //0-7 (internal)
	iunit = addr.getFedFeUnit();
	//0-11 (internal)
	ichan = addr.getFeUnitChannel();
	//0-95 (internal)
	chan = 12*( iunit ) + ichan;
      } catch(...) { 
	rawToDigi_.unpacker()->handleException( __func__, "Problem using Fed9UAddress" ); 
      } 
      
      try {
	// temporary check to find corrupted FED data
        #ifdef USE_PATCH_TO_CATCH_CORRUPT_FED_DATA
	uint16_t last_strip = 0;
	uint16_t strips = 256 * iconn->nApvPairs();
        #endif
	Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvents_[iconn->fedId()]->channel( iunit, ichan )).getIterator();
	for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0;) {
	  uint16_t first_strip = iconn->apvPairNumber()*256 + *i++;
	  unsigned char width = *i++; 
	  for ( uint16_t istr = 0; istr < ((uint16_t)width); istr++) {
	    uint16_t strip = first_strip + istr;
	    // temporary check to find corrupted FED data
            #ifdef USE_PATCH_TO_CATCH_CORRUPT_FED_DATA
	    if ( !( strip < strips && ( !strip || strip > last_strip ) ) ) { 
	      edm::LogWarning(mlRawToDigi_)
		<< "[OldSiStripRawToClustersLazyUnpacker::" 
		<< __func__ << "]"
		<< " Corrupt FED data found for FED id " 
		<< iconn->fedId()
		<< " and channel " 
		<< iconn->fedCh()
		<< "!  present strip: " 
		<< strip
		<< "  last strip: " 
		<< last_strip
		<< "  detector strips: " 
		<< strips;
	      continue; 
	    }
	    last_strip = strip;
            #endif	 
	    clusterizer_->algorithm()->add(record,idet->first,strip,(uint16_t)(*i++));
	  }
	}
      } catch(...) { 
	std::stringstream sss;
	sss << "Problem accessing ZERO_SUPPR data for FED id/ch: " 
	    << iconn->fedId() 
	    << "/" 
	    << chan;
	rawToDigi_.unpacker()->handleException( __func__, sss.str() ); 
      } 
    }
    clusterizer_->algorithm()->endDet(record,idet->first);
  }
}
  
namespace sistrip { 

  RawToClustersLazyUnpacker::RawToClustersLazyUnpacker(const SiStripRegionCabling& regioncabling, StripClusterizerAlgorithm& clustalgo, SiStripRawProcessingAlgorithms& rpAlgos, const FEDRawDataCollection& data, bool dump) :

    raw_(&data),
    regions_(&(regioncabling.getRegionCabling())),
    clusterizer_(&clustalgo),
    rawAlgos_(&rpAlgos),
    buffers_(),
    rawToDigi_(0,0,0,0,0),
    dump_(dump),
    mode_(sistrip::READOUT_MODE_INVALID),
    fedRawData_()
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
      if ( !(idet->first) || (idet->first == sistrip::invalid32_) || !clusterizer_->stripByStripBegin(idet->first)) { continue; }
    
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
	  const FEDRawData& input = raw_->FEDData( static_cast<int>(fedId) );
	
	  // Cache new correctly-ordered FEDRawData object (to maintain scope for Fed9UEvent)
	  fedRawData_.push_back( FEDRawData() );
	  rawToDigi_.locateStartOfFedBuffer( fedId, input, fedRawData_.back() );
	
	  // Check on FEDRawData pointer
	  if ( !fedRawData_.back().data() ) {
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
	  if ( !fedRawData_.back().size() ) {
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
            buffer = new sistrip::FEDBuffer(fedRawData_.back().data(),fedRawData_.back().size());
            if (!buffer->doChecks()) throw cms::Exception("FEDBuffer") << "FED Buffer check fails for FED ID" << fedId << ".";
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
	    rawToDigi_.dumpRawData( fedId, input, ss );
	    LogTrace(mlRawToDigi_) 
	      << ss.str();
	  }

	  // record readout mode
	  mode_ = buffer->readoutMode();
	}

	// check channel
        const uint8_t fedCh = iconn->fedCh();
	if (!buffer->channelGood(fedCh)) {
          if (edm::isDebugEnabled()) {
            std::ostringstream ss;
            ss << "Problem unpacking channel " << fedCh << " on FED " << fedId;
            edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
          }
          continue;
        }
      
	// Determine APV std::pair number
	uint16_t ipair = iconn->apvPairNumber();


	if (mode_ == sistrip::READOUT_MODE_ZERO_SUPPRESSED ) { 
	
	  // create unpacker
	  sistrip::FEDZSChannelUnpacker unpacker = sistrip::FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(buffer->channel(fedCh));
	
	  // unpack
	  while (unpacker.hasData()) {
	    clusterizer_->stripByStripAdd(unpacker.sampleNumber()+ipair*256,unpacker.adc(),record);
	    unpacker++;
	  }
	}

	else if (mode_ == sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE ) { 
	
	  // create unpacker
	  sistrip::FEDZSChannelUnpacker unpacker = sistrip::FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(buffer->channel(fedCh));
	
	  // unpack
	  while (unpacker.hasData()) {
	    clusterizer_->stripByStripAdd(unpacker.sampleNumber()+ipair*256,unpacker.adc(),record);
	    unpacker++;
	  }
	}

	else if (mode_ == sistrip::READOUT_MODE_VIRGIN_RAW ) {

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
	  rawAlgos_->subtractorPed->subtract( id, ipair*256, digis);
	  rawAlgos_->subtractorCMN->subtract( id, digis);
	  edm::DetSet<SiStripDigi> zsdigis(id);
	  rawAlgos_->suppressor->suppress( digis, zsdigis);
	  for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
	    clusterizer_->stripByStripAdd( it->strip(), it->adc(), record);
	  }
	}

	else if (mode_ == sistrip::READOUT_MODE_PROC_RAW ) {

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
	  rawAlgos_->subtractorCMN->subtract( id, digis);
	  edm::DetSet<SiStripDigi> zsdigis(id);
	  rawAlgos_->suppressor->suppress( digis, zsdigis);
	  for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
	    clusterizer_->stripByStripAdd( it->strip(), it->adc(), record);
	  }
	}

	else {
	  edm::LogWarning(sistrip::mlRawToCluster_)
	    << "[sistrip::RawToClustersLazyGetter::" 
	    << __func__ << "]"
	    << " FEDRawData readout mode "
	    << mode_
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
