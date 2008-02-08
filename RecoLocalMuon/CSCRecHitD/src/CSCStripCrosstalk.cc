
// Extract XTalks
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripCrosstalk.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <CondFormats/CSCObjects/interface/CSCDBCrosstalk.h>
#include <CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h>
#include <DataFormats/MuonDetId/interface/CSCIndexer.h>

#include <map>
#include <vector>
#include <iostream>

CSCStripCrosstalk::CSCStripCrosstalk( const edm::ParameterSet & ps ) {

  debug                  = ps.getUntrackedParameter<bool>("CSCDebug");  

  theIndexer = new CSCIndexer;

}


CSCStripCrosstalk::~CSCStripCrosstalk() {

  delete theIndexer;
}


/* getCrossTalk
 *
 */
void CSCStripCrosstalk::getCrossTalk( const CSCDetId& id, int centralStrip, std::vector<float>& xtalks) {

  //  CSCIndexer* theIndexer = new CSCIndexer;

  xtalks.clear();

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

  float m_left = 0.;
  float b_left = 0.;
  float m_right = 0.;
  float b_right = 0.;
  int idx = 0;


  // Cluster of 3 strips, so get x-talks for these 3 strips
  for ( int i = 0; i < 3; ++i ) {

    LongIndexType sid = sid2[i] -1;   // DB starts at zero, whereas indexer starts at 1
      
    m_left  = xTalk_->crosstalk[sid].xtalk_slope_left;
    b_left  = xTalk_->crosstalk[sid].xtalk_intercept_left;
    m_right = xTalk_->crosstalk[sid].xtalk_slope_right;
    b_right = xTalk_->crosstalk[sid].xtalk_intercept_right;

    // std::cout << "X-talks are " << m_left << " " << b_left << " " << m_right << " " << b_right << std::endl;

    xtalks.push_back(m_left);
    xtalks.push_back(b_left);
    xtalks.push_back(m_right);
    xtalks.push_back(b_right);
    idx++;
  }
}
