// Last commit: $Id: test_PedestalsBuilder.cc,v 1.5 2013/05/30 21:52:09 gartung Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/test_PedestalsBuilder.cc,v $

#include "OnlineDB/SiStripESSources/test/stubs/test_PedestalsBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
void test_PedestalsBuilder::analyze( const edm::Event& event, const edm::EventSetup& setup ) {
  
  LogTrace(mlCabling_) 
    << "[test_PedestalsBuilder::" << __func__ << "]"
    << " Dumping all FED connections...";
  
  edm::ESHandle<SiStripPedestals> peds;
  setup.get<SiStripPedestalsRcd>().get( peds ); 
  
  // Retrieve DetIds in Pedestals object
  vector<uint32_t> det_ids;
  peds->getDetIds( det_ids );
  
  // Iterate through DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {

    // Retrieve pedestals for given DetId
    SiStripPedestals::Range range = peds->getRange( *det_id );

    // Check if module has 512 or 768 strips (horrible!)
    uint16_t nstrips = 2*sistrip::STRIPS_PER_FEDCH;
//     try {
//       peds->getPed( 2*sistrip::STRIPS_PER_FEDCH, range );
//     } catch ( cms::Exception& e ) {
//       nstrips = 2*sistrip::STRIPS_PER_FEDCH;
//     }

    stringstream ss;
    ss << "[test_PedestalsBuilder::" << __func__ << "]"
       << " Found " << nstrips
       << " pedestals for DetId " << *det_id
       << " (ped/low/high): ";

    // Extract peds and low/high thresholds
    for ( uint16_t istrip = 0; istrip < nstrips; istrip++ ) {
      ss << peds->getPed( istrip, range ) << ", ";
    }

    LogTrace(mlCabling_) << ss.str();
    
  }
  
}

