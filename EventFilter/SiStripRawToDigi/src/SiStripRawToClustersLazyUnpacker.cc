#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersLazyUnpacker.h"

//Data Formats
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <sstream>
#include <iostream>

//stl
#include <vector>

using namespace sistrip;

SiStripRawToClustersLazyUnpacker::SiStripRawToClustersLazyUnpacker(const SiStripRegionCabling& regioncabling, const SiStripClusterizerFactory& clustfact, const FEDRawDataCollection& data) :

  edm::SiStripLazyUnpacker<SiStripCluster>(regioncabling.getRegionCabling().size()),
  raw_(&data),
  regions_(&(regioncabling.getRegionCabling())),
  clusterizer_(&clustfact),
  fedEvents_(),
  rawToDigi_(0,0,0,0,0)

{
  fedEvents_.assign(1024,static_cast<Fed9U::Fed9UEvent*>(0));
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


void SiStripRawToClustersLazyUnpacker::fill(uint32_t& iregion) {
 
  //Get region cabling and record
  const SiStripRegionCabling::RegionMap& rmap = (*regions_)[iregion];

  //Loop regions dets
  SiStripRegionCabling::RegionMap::const_iterator idet = rmap.begin();
  for (;idet!=rmap.end();idet++) {
    
    //If det id is null or invalid continue.
    if ( !(idet->first) || (idet->first == sistrip::invalid32_) ) { continue; }
    
    //Loop over apv-pairs of det
    const std::vector<FedChannelConnection>& conns = idet->second;
    for (uint16_t ipair = 0; ipair < conns.size(); ipair++) {
    
      //Get FedChannelConnection
      const FedChannelConnection& conn = conns[ipair];    
      uint16_t fedId = conn.fedId();
      
      //If Fed hasnt already been initialised, extract data and initialise
      if (!fedEvents_[fedId]) {
	
	// Retrieve FED raw data for given FED 
	const FEDRawData& input = raw_->FEDData( static_cast<int>(fedId) );
	
	// Recast data to suit Fed9UEvent
	Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( input.data() ) );
	Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( input.size() / 4 ); 
	
	// Check on FEDRawData pointer
	if ( !data_u32 ) {
	  edm::LogWarning(mlRawToCluster_)
	    << "[SiStripRawToClustersLazyGetter::" 
	    << __func__ 
	    << "]"
	    << " NULL pointer to FEDRawData for FED id " << fedId;
	  continue;
	}	
	
	// Check on FEDRawData size
	if ( !size_u32 ) {
	  edm::LogWarning(mlRawToCluster_)
	    << "[SiStripRawToClustersLazyGetter::" << __func__ << "]"
	    << " FEDRawData has zero size for FED id " << fedId;
	  continue;
	}
	
	// Construct Fed9UEvent using present FED buffer
	try {
	  fedEvents_[fedId] = new Fed9U::Fed9UEvent(data_u32,0,size_u32);
	} catch(...) { rawToDigi_.handleException( __func__, "Problem when constructing Fed9UEvent" ); }
	
	/*
	//Check Fed9UEvent
	try {
	//fedEvents_[fedId]->checkEvent(); 
	} catch(...) { rawToDigi_.handleException( __func__, "Problem when checking Fed9UEventStreamLine" ); }
	*/
	
	/*
	// Retrieve readout mode
	sistrip::FedReadoutMode mode = sistrip::UNDEFINED_FED_READOUT_MODE;
	try {
	mode = rawToDigi_.fedReadoutMode( static_cast<unsigned int>( fedEvents_[fedId]->getSpecialTrackerEventType() ) );
	} catch(...) { rawToDigi_.handleException( __func__, "Problem extracting readout mode from Fed9UEvent" ); } 
	
	if ( mode != sistrip::ZERO_SUPPR ) { 
	edm::LogWarning(sistrip::mlRawToCluster_)
	<< "[SiStripRawClustersLazyGetter::" << __func__ << "]"
	<< " Readout mode for FED id " << fedId
	<< " not zero suppressed.";
	continue;
	}
	*/
	
	/*
	// Dump of FEDRawData to stdout
	if ( dumpFrequency_ && !(event.id().event()%dumpFrequency_) ) {
	stringstream ss;
	rawToDigi_.dumpRawData( fed_id, input, ss );
	LogTrace(mlRawToDigi_) << ss.str();
	}
	*/
	
      }
      
      //Calculate corresponding FED unit, channel
      Fed9U::Fed9UAddress addr;
      uint16_t iunit = 0;
      uint16_t ichan = 0;
      uint16_t chan = 0;
      try {
	addr.setFedChannel( static_cast<unsigned char>( conn.fedCh() ) );
	//0-7 (internal)
	iunit = addr.getFedFeUnit();/*getExternalFedFeUnit() for StreamLine*/
	//0-11 (internal)
	ichan = addr.getFeUnitChannel();/*getExternalFeUnitChannel()*/
	//0-95 (internal)
	chan = 12*( iunit ) + ichan;
      } catch(...) { 
	rawToDigi_.handleException( __func__, "Problem using Fed9UAddress" ); 
      } 
      
      try{ 
	
	Fed9U::Fed9UEventIterator fed_iter = const_cast<Fed9U::Fed9UEventChannel&>(fedEvents_[fedId]->channel( iunit, ichan )).getIterator();
	
	for (Fed9U::Fed9UEventIterator i = fed_iter+7; i.size() > 0;) {
	  uint16_t strip = ipair*256 + *i++;
	  unsigned char width = *i++;       // cluster width in strips 
	  for ( uint16_t istr = 0; istr < ((uint16_t)width); istr++) {
	    clusterizer_->algorithm()->add(record(),idet->first,(uint16_t)(strip+istr),(uint16_t)(*i++));
	  }
	}
	
      } catch(...) { 
	std::stringstream sss;
	sss << "Problem accessing ZERO_SUPPR data for FED id/ch: " 
	    << fedId << "/" << chan;
	rawToDigi_.handleException( __func__, sss.str() ); 
      } 
    }
    clusterizer_->algorithm()->endDet(record(),idet->first);
  }
}
