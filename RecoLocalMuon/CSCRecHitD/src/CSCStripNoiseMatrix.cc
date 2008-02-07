// Read in strip digi collection and apply calibrations to ADC counts

#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripNoiseMatrix.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h>
#include <CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>

#include "DataFormats/MuonDetId/interface/CSCIndexer.h"


#include <map>
#include <vector>

CSCStripNoiseMatrix::CSCStripNoiseMatrix( const edm::ParameterSet & ps ) {
  debug                  = ps.getUntrackedParameter<bool>("CSCDebug");  

  theIndexer = new CSCIndexer;

}


CSCStripNoiseMatrix::~CSCStripNoiseMatrix() {

  delete theIndexer;
}


/* getNoiseMatrix
 *
 */
  void CSCStripNoiseMatrix::getNoiseMatrix( const CSCDetId& id, int centralStrip, std::vector<float>& nMatrix ) {

  //  CSCIndexer* theIndexer = new CSCIndexer;

  nMatrix.clear();

  // Initialize values in case can't find chamber with constants
  // These are ME1/2 constants...
  float elem[15];
  elem[0] = 8.64;
  elem[1] = 3.47;
  elem[2] = 2.45;
  elem[3] = 8.60;
  elem[4] = 3.28;
  elem[5] = 1.88;
  elem[6] = 8.61;
  elem[7] = 3.18;
  elem[8] = 1.99;
  elem[9] = 7.67;
  elem[10] = 2.64;
  elem[11] = 0.;
  elem[12] = 7.71;
  elem[13] = 0.;
  elem[14] = 0.;


  IndexType strip1 = centralStrip;

  std::vector<LongIndexType> sid2;

  // Compute channel id used for retrieving gains from database
  int rg = id.ring();

  bool isME1a = false;

  // Note that ME1/a constants are stored in ME1/1 which also constains ME1/b
  // ME1/a constants are stored beginning at entry 64
  // Also, only have only 16 strips to worry about for ME1a
  if ( id.ring() == 4 ) {
    isME1a = true;
    rg = 1;
    strip1 = centralStrip%16;
    if (strip1 == 0) strip1 = 16;
    strip1 = strip1 + 64;
    const CSCDetId testId( id.endcap(), 1, 1, id.chamber(), id.layer() );
    LongIndexType idDB = theIndexer->stripChannelIndex( testId, strip1);
    
    if (strip1 == 65) {
      sid2.push_back(idDB+15);
      sid2.push_back(idDB);   
      sid2.push_back(idDB+1); 
    } else if (strip1 == 80) {
      sid2.push_back(idDB-1); 
      sid2.push_back(idDB);   
      sid2.push_back(idDB-15);
    } else {
      sid2.push_back(idDB-1);
      sid2.push_back(idDB);  
      sid2.push_back(idDB+1);
    }
  } else {
    LongIndexType idDB = theIndexer->stripChannelIndex( id, strip1);
    sid2.push_back(idDB-1);
    sid2.push_back(idDB);  
    sid2.push_back(idDB+1);
  }


  for ( int i = 0; i < 3; ++i) {
  
    LongIndexType sid = sid2[i] -1;    // DB starts at zero, whereas indexer starts at 1

    float w = getStripGain( sid );
    w = w*w;
     
    elem[0] = Noise->matrix[sid].elem33 * w;
    elem[1] = Noise->matrix[sid].elem34 * w;
    elem[2] = Noise->matrix[sid].elem35 * w;
    elem[3] = Noise->matrix[sid].elem44 * w;
    elem[4] = Noise->matrix[sid].elem45 * w;
    elem[5] = Noise->matrix[sid].elem46 * w;
    elem[6] = Noise->matrix[sid].elem55 * w; 
    elem[7] = Noise->matrix[sid].elem56 * w; 
    elem[8] = Noise->matrix[sid].elem57 * w; 
    elem[9] = Noise->matrix[sid].elem66 * w; 
    elem[10]= Noise->matrix[sid].elem67 * w;
    elem[11]= 0.; 
    elem[12]= Noise->matrix[sid].elem77 * w;
    elem[13]= 0.;
    elem[14]= 0.;
      
    // Test that elements make sense:
    bool isFlawed = false;      
    for ( int k = 0; k < 15; ++k) {
      // make sure the number isn't too close to zero...
      if (elem[k] < 0.001) elem[k] = 0.001;
      // make sure no number isn't too big...
      if (elem[k] > 50.) isFlawed = true; 
    }

    if ( isFlawed ) {
      // These are fake ME1/2:
      elem[0] = 8.64;
      elem[1] = 3.47;
      elem[2] = 2.45;
      elem[3] = 8.60;
      elem[4] = 3.28;
      elem[5] = 1.88;
      elem[6] = 8.61;
      elem[7] = 3.18;
      elem[8] = 1.99;
      elem[9] = 7.67;
      elem[10] = 2.64;
      elem[11] = 0.;
      elem[12] = 7.71;
      elem[13] = 0.;
      elem[14] = 0.;
    }

    for (int k = 0; k < 15; ++k) nMatrix.push_back( elem[k] );
  }
}


/* getStripWeights
 *
 */
float CSCStripNoiseMatrix::getStripGain( LongIndexType thisStrip ) {

  float w = 1.0;
  w = globalGainAvg/Gains->gains[thisStrip].gain_slope;

  if ( w < 0.5 ) w = 0.5;
  if ( w > 1.5 ) w = 1.5;
  return w;

}
