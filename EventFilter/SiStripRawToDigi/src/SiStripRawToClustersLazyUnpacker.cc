#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersLazyUnpacker.h"
#include <sstream>
#include <iostream>

using namespace sistrip;

SiStripRawToClustersLazyUnpacker::SiStripRawToClustersLazyUnpacker(const SiStripRegionCabling& regioncabling, const SiStripClusterizerFactory& clustfact, const FEDRawDataCollection& data) :

  raw_(&data),
  regions_(&(regioncabling.getRegionCabling())),
  clusterizer_(&clustfact),
  fedEvents_(),
  fedModes_(),
  rawToDigi_(0,0,0,0,0)

{
  fedEvents_.assign(1024,static_cast<Fed9U::Fed9UEvent*>(0));
  fedModes_.assign(1024,sistrip::UNDEFINED_FED_READOUT_MODE);
}

SiStripRawToClustersLazyUnpacker::~SiStripRawToClustersLazyUnpacker() {
  
  std::vector< Fed9U::Fed9UEvent*>::iterator ifedevent = fedEvents_.begin();
  for (; ifedevent!=fedEvents_.end(); ifedevent++) {
    if (*ifedevent) {
      delete (*ifedevent);
      *ifedevent = 0;
    }
  }
}

void SiStripRawToClustersLazyUnpacker::fill(const uint32_t& index, record_type& record) {

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
    if ( !(idet->first) || (idet->first == sistrip::invalid32_) ) { continue; }
    
    // Loop over apv-pairs of det
    std::vector<FedChannelConnection>::const_iterator iconn = idet->second.begin();
    for (;iconn!=idet->second.end();iconn++) {
      
      // If fed id is null or connection is invalid continue
      if ( !iconn->fedId() || !iconn->isConnected() ) { continue; }    
      
      // If Fed hasnt already been initialised, extract data and initialise
      if (!fedEvents_[iconn->fedId()]) {
	
	// Retrieve FED raw data for given FED
	const FEDRawData& input = raw_->FEDData( static_cast<int>(iconn->fedId()) );
	
	// @@ TEMP FIX DUE TO FED SW AND DAQ INCOMPATIBLE FORMATS (32-BIT WORD SWAPPED)
	FEDRawData& temp = const_cast<FEDRawData&>( input ); 
	FEDRawData output;
	rawToDigi_.locateStartOfFedBuffer( iconn->fedId(), temp, output );
	temp.resize( output.size() );
	memcpy( temp.data(), output.data(), output.size() ); //@@ edit event data!!!

	// Recast data to suit Fed9UEvent
	Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( input.data() ) );
	Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( input.size() / 4 ); 
	
	// Check on FEDRawData pointer
	if ( !data_u32 ) {
	  if ( edm::isDebugEnabled() ) {
	    edm::LogWarning(mlRawToCluster_)
	      << "[SiStripRawToClustersLazyGetter::" 
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
	    edm::LogWarning(mlRawToCluster_)
	      << "[SiStripRawToClustersLazyGetter::" 
	      << __func__ << "]"
	      << " FEDRawData has zero size for FED id " 
	      << iconn->fedId();
	  }
	  continue;
	}
	
	// Construct Fed9UEvent using present FED buffer
	try {
	  fedEvents_[iconn->fedId()] = new Fed9U::Fed9UEvent(data_u32,0,size_u32);
	} catch(...) { 
	  rawToDigi_.handleException( __func__, "Problem when constructing Fed9UEvent" ); 
	  if ( fedEvents_[iconn->fedId()] ) { delete fedEvents_[iconn->fedId()]; }
	  fedEvents_[iconn->fedId()] = 0;
	  fedModes_[iconn->fedId()] = sistrip::UNDEFINED_FED_READOUT_MODE;
	  continue;
	}
	
	/*
	// Check Fed9UEvent
	try {fedEvents_[iconn->fedId()]->checkEvent();} 
	catch(...) {rawToDigi_.handleException( __func__, "Problem when checking Fed9UEventStreamLine" );}
	*/
	
	// Retrieve readout mode
	try {fedModes_[iconn->fedId()] = rawToDigi_.fedReadoutMode( static_cast<unsigned int>( fedEvents_[iconn->fedId()]->getSpecialTrackerEventType() ) );} 
	catch(...) {rawToDigi_.handleException( __func__, "Problem extracting readout mode from Fed9UEvent" );} 
      }
      
      // Check readout mode is ZERO_SUPPRESSED or ZERO_SUPPRESSED_LITE
      if (fedModes_[iconn->fedId()] != sistrip::FED_ZERO_SUPPR && fedModes_[iconn->fedId()] != sistrip::FED_ZERO_SUPPR_LITE) { 
	edm::LogWarning(sistrip::mlRawToCluster_)
	  << "[SiStripRawClustersLazyGetter::" 
	  << __func__ 
	  << "]"
	  << " Readout mode for FED id " 
	  << iconn->fedId()
	  << " not zero-suppressed or zero-suppressed lite.";
	continue;
      }
      
      // Calculate corresponding FED unit, channel
      uint16_t iunit = 0, ichan = 0, chan = 0;
      try {
	Fed9U::Fed9UAddress addr;
	addr.setFedChannel( static_cast<unsigned char>(iconn->fedCh()));
	iunit = addr.getFedFeUnit(); //0-7 (internal)
	ichan = addr.getFeUnitChannel(); //0-11 (internal)
	chan = 12*( iunit ) + ichan; //0-95 (internal)
      } catch(...) { 
	rawToDigi_.handleException(__func__, "Problem using Fed9UAddress"); 
      } 
      
      try {
#ifdef USE_PATCH_TO_CATCH_CORRUPT_FED_DATA
	uint16_t last_strip = 0;
	uint16_t strips = 256 * iconn->nApvPairs();
#endif
	
	Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvents_[iconn->fedId()]->channel( iunit, ichan )).getIterator();
	Fed9U::Fed9UEventIterator i = fed_iter+(fedModes_[iconn->fedId()] == sistrip::FED_ZERO_SUPPR ? 7 : 2);
	for (;i.size() > 0;) {
	  uint16_t first_strip = iconn->apvPairNumber()*256 + *i++;
	  unsigned char width = *i++; 
	  for ( uint16_t istr = 0; istr < ((uint16_t)width); istr++) {
	    uint16_t strip = first_strip + istr;

#ifdef USE_PATCH_TO_CATCH_CORRUPT_FED_DATA
	    if ( !( strip < strips && ( !strip || strip > last_strip ) ) ) { // check for corrupt FED data
	      if ( edm::isDebugEnabled() ) {
		std::stringstream ss;
		ss << "[SiStripRawToDigiUnpacker::" << __func__ << "]"
		   << " Corrupt FED data found for FED id " << iconn->fedId()
		   << " and channel " << iconn->fedCh()
		   << "!  present strip: " << strip
		   << "  last strip: " << last_strip
		   << "  detector strips: " << strips;
		edm::LogWarning(mlRawToDigi_) << ss.str();
		}
	      continue; 
	    } 
	    last_strip = strip;
#endif

	    clusterizer_->algorithm()->add(record,idet->first,strip,(uint16_t)(*i++));
	  }
	}	
      } catch(...) { 
	std::stringstream sss;
	sss << "Problem accessing data for FED id/ch: " 
	    << iconn->fedId() 
	    << "/" 
	    << chan;
	rawToDigi_.handleException( __func__, sss.str() ); 
      } 
    }
    clusterizer_->algorithm()->endDet(record,idet->first);
  }
}
  
