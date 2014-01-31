#ifndef GEMValidation_TrackMatcher_h
#define GEMValidation_TrackMatcher_h

/**\class TrackMatcher

 Description: Matching of tracks to SimTrack

 Original Author:  "Sven Dildick"
 $Id: $
*/

#include "GEMCode/GEMValidation/src/CSCDigiMatcher.h"
#include "GEMCode/GEMValidation/src/TFTrack.h"
#include "GEMCode/GEMValidation/src/TFCand.h"
#include "GEMCode/GEMValidation/src/GMTRegCand.h"
#include "GEMCode/GEMValidation/src/GMTCand.h"
#include "GEMCode/GEMValidation/src/L1Extra.h"

class TrackMatcher : public CSCStubMatcher
{
 public:
  TrackMatcher();
  ~TrackMatcher();

  //  std::vector<TFTrack>& tfTrack();
  //  TFTrack* bestTFTrack(bool sortPtFirst=1);
  //  std::vector<TFCand>& tfCand();
  //  TFCand* bestTFCand(bool sortPtFirst=1);
  //  std::vector<GMTRegCCand>& gmtRegCand();
  //  GMTRegCand* bestGmtRegcand(bool sortPtFirst=1);
  //  std::vector<GMTCCand>& gmtCand();
  //  GMTCand* bestGmtcand(bool sortPtFirst=1);
  //  std::vector<GMTCCand>& gmtCand();
  //  GMTCand* bestGmtcand(bool sortPtFirst=1);

  
 private:
};

#endif
