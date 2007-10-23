#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDRawDataCheck.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "interface/shared/fed_trailer.h"
#include "boost/cstdint.hpp"
#include <sstream>
#include <string>
#include <vector>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripFEDRawDataCheck::SiStripFEDRawDataCheck( const edm::ParameterSet& pset ) 
  : label_( pset.getUntrackedParameter<std::string>("ProductLabel","source") ),
    instance_( pset.getUntrackedParameter<std::string>("ProductInstance","") ),
    unpacker_( new SiStripRawToDigiUnpacker(0,0,0,0,0) )
{
  LogTrace(mlRawToDigi_)
    << "[SiStripFEDRawDataCheck::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
SiStripFEDRawDataCheck::~SiStripFEDRawDataCheck() 
{
  LogTrace(mlRawToDigi_)
    << "[SiStripFEDRawDataCheck::" << __func__ << "]"
    << " Destructing object...";
  if ( unpacker_ ) { delete unpacker_; }
}

// -----------------------------------------------------------------------------
// 
void SiStripFEDRawDataCheck::analyze( const edm::Event& event,
				      const edm::EventSetup& setup ) {
  
  // Retrieve FED cabling object
  edm::ESHandle<SiStripFedCabling> cabling;
  setup.get<SiStripFedCablingRcd>().get( cabling );
  
  // Retrieve FEDRawData collection
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( label_, instance_, buffers ); 
  
  // Local cache of fed ids from cabling
  std::vector<uint16_t> fed_ids = cabling->feds();
  
  // Containers storing fed ids and buffer sizes
  typedef std::pair<uint16_t,uint16_t> Fed;
  typedef std::vector<Fed> Feds;
  Feds trg_feds; // trigger feds
  Feds trk_feds; // tracker feds
  Feds non_trk;  // feds from other sub-dets
  Feds in_data;  // feds in data but missing from cabling
  Feds in_cabl;  // feds in cabling but missing from data
  
  for ( uint16_t ifed = 0; ifed <= sistrip::CMS_FED_ID_MAX; ++ifed ) {
    const FEDRawData& fed = buffers->FEDData( static_cast<int>(ifed) );
    uint8_t* data = const_cast<uint8_t*>( fed.data() ); // look for trigger fed
    fedt_t* fed_trailer = reinterpret_cast<fedt_t*>( data + fed.size() - sizeof(fedt_t) );
    if ( fed_trailer->conscheck == 0xDEADFACE ) { trg_feds.push_back(Fed(ifed,fed.size())); }
    else {
      if ( ifed < sistrip::FED_ID_MIN ||
	   ifed > sistrip::FED_ID_MAX ) {
	if ( fed.size() ) { non_trk.push_back(Fed(ifed,fed.size())); }
      } else { 
	bool found = std::find( fed_ids.begin(), fed_ids.end(), ifed ) != fed_ids.end();
	if ( fed.size() ) {
	  if ( found ) { trk_feds.push_back(Fed(ifed,fed.size())); }
	  else { in_data.push_back(Fed(ifed,fed.size())); }
	}
	if ( found && fed.size() == 0 ) { in_cabl.push_back(Fed(ifed,0)); }
      }	
    }
  }

  // Trigger FEDs
  {
    std::stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << trg_feds.size() 
       <<  " trigger FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < trg_feds.size(); ++ifed ) {
      ss << trg_feds[ifed].first << "(" << trg_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Not strip tracker FEDs
  {
    std::stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << non_trk.size() 
       <<  " FEDs from other sub-dets";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < non_trk.size(); ++ifed ) {
      ss << non_trk[ifed].first << "(" << non_trk[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // Strip tracker FEDs
  {
    std::stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << trk_feds.size() 
       <<  " strip tracker FEDs";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < trk_feds.size(); ++ifed ) {
      ss << trk_feds[ifed].first << "(" << trk_feds[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }
  
  // FEDs in data but missing from cabling
  {
    std::stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << in_data.size() 
       <<  " strip tracker FEDs in data but missing from cabling";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and buffer sizes [bytes]): ";
    for ( uint16_t ifed = 0; ifed < in_data.size(); ++ifed ) {
      ss << in_data[ifed].first << "(" << in_data[ifed].second << ") ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }

  // FEDs in cabling but missing from data 
  {
    std::stringstream ss;
    ss << "[SiStripFEDRawDataCheck::" << __func__ << "]"
       << " Found " << in_cabl.size() 
       <<  " strip tracker FEDs in cabling but missing from data";
    edm::LogVerbatim(mlRawToDigi_) << ss.str();
    ss << " with ids (and zero buffer size): ";
    for ( uint16_t ifed = 0; ifed < in_cabl.size(); ++ifed ) {
      ss << in_cabl[ifed].first << " ";
    }
    LogTrace(mlRawToDigi_) << ss.str();
  }
  
//   // Analyze strip tracker FED buffers in data
//   for ( uint16_t ifed = 0; ifed < trk_feds.size(); ++ifed ) {
//     const FEDRawData& fed = buffers->FEDData( static_cast<int>(ifed) );
    
//     // Remove extra header words and check if 32-bit word swapped
//     FEDRawData output; 
//     if ( unpacker_ ) { unpacker_->locateStartOfFedBuffer( ifed, fed, output ); }
    
//     // Initialise Fed9UEvent using present FED buffer
//     Fed9U::u32* data_u32 = reinterpret_cast<Fed9U::u32*>( const_cast<unsigned char*>( output.data() ) );
//     Fed9U::u32  size_u32 = static_cast<Fed9U::u32>( output.size() / 4 ); 
//     try {
//       fedEvent_->Init( data_u32, 0, size_u32 ); 
//       //fedEvent_->checkEvent();
//     } catch(...) { 
//       std::string temp = "Problem when creating Fed9UEvent for FED id " + ifed; 
//       handleException( __func__, temp ); 
//       continue;
//     } 

//     try {
//       fedEvent_->checkEvent();
//     } catch(...) { 
//       std::string temp = "Problem when creating Fed9UEvent for FED id " + ifed; 
//       handleException( __func__, temp ); 
//       continue;
//     } 
  
}
