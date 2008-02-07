// Read in strip digi collection and apply calibrations to ADC counts


#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripGain.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

#include <map>
#include <vector>
#include <iostream>

CSCStripGain::CSCStripGain( const edm::ParameterSet & ps ) {

  debug                  = ps.getUntrackedParameter<bool>("CSCDebug");  
  //isData                 = ps.getUntrackedParameter<bool>("CSCIsRunningOnData");

  theIndexer = new CSCIndexer;

}


CSCStripGain::~CSCStripGain() {

  delete theIndexer;
}


/* getStripWeights
 *
 */
void CSCStripGain::getStripGain( const CSCDetId& id, float* weights ) {

  //  CSCIndexer* theIndexer = new CSCIndexer;


  // Compute channel id used for retrieving gains from database
  int st = id.station();
  int rg = id.ring();

  unsigned strip1 = 1;
  unsigned nStrips = 80;
  if ( st == 1 && rg == 1) nStrips = 64;
  if ( st == 1 && rg == 3) nStrips = 64;

  // Note that ME1/a constants are stored in ME1/1 (ME1/b) starting at entry 65
  if ( rg == 4 ) {
    const CSCDetId testId( id.endcap(), 1, 1, id.chamber(), id.layer() );
    strip1 = 65;
    LongIndexType idDB = theIndexer->stripChannelIndex( testId, strip1);

    for ( unsigned i = 0; i < 16; ++i) {
      LongIndexType sid = idDB + i -1;                         // DB start at 0, indexer start at 1

      float w = globalGainAvg/Gains_->gains[sid].gain_slope;

      if (w > 1.5) w = 1.5;
      if (w < 0.5) w = 0.5;

      weights[i]    = w;
      weights[i+16] = w;
      weights[i+32] = w;
    }
  } 
  // All other chamber types
  else {
    
    LongIndexType idDB = theIndexer->stripChannelIndex( id, strip1);

    for ( unsigned i = 0; i < nStrips; ++i) {
      LongIndexType sid = idDB + i -1;                       // DB start at 0, indexer start at 1

      float w = globalGainAvg/Gains_->gains[sid].gain_slope;
      if (w > 1.5) w = 1.5;
      if (w < 0.5) w = 0.5;
      weights[i]    = w;
    }
  }
}
