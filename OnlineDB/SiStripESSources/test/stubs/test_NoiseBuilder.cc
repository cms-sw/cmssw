// Last commit: $Id: test_NoiseBuilder.cc,v 1.5 2013/05/30 21:52:09 gartung Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/test_NoiseBuilder.cc,v $

#include "OnlineDB/SiStripESSources/test/stubs/test_NoiseBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include <iostream>
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
void test_NoiseBuilder::analyze(const edm::Event& event, const edm::EventSetup& setup ) {
  
  LogTrace(mlCabling_) 
    << "[test_NoiseBuilder::" << __func__ << "]"
    << " Dumping all FED connections...";
  
  edm::ESHandle<SiStripNoises> noise;
  setup.get<SiStripNoisesRcd>().get( noise ); 
  
  // Retrieve DetIds in Noise object
  vector<uint32_t> det_ids;
  noise->getDetIds( det_ids );
  
  // Iterate through DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {

    // Retrieve noise for given DetId
    SiStripNoises::Range range = noise->getRange( *det_id );

    // Check if module has 512 or 768 strips (horrible!)
    uint16_t nstrips = 2*sistrip::STRIPS_PER_FEDCH;
//     try {
//       noise->getNoise( 2*sistrip::STRIPS_PER_FEDCH, range );
//     } catch ( cms::Exception& e ) {
//       nstrips = 2*sistrip::STRIPS_PER_FEDCH;
//     }

    stringstream ss;
    ss << "[test_NoiseBuilder::" << __func__ << "]"
       << " Found " << nstrips
       << " noise for DetId " << *det_id
       << " (noise/disabled): ";

    // Extract noise and low/high thresholds
    for ( uint16_t istrip = 0; istrip < nstrips; istrip++ ) {
      ss << noise->getNoise( istrip, range ) << "/"
	 << ", ";
    }

    LogTrace(mlCabling_) << ss.str();
    
  }
  
}

