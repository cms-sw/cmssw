#include "RecoMuon/Navigation/interface/MuonNavigableLayer.h"
//   Ported from ORCA.
//   New methods compatibleLayers are added.
//   $Date: 2006/04/01 21:50:28 $
//   $Revision: 1.1 $

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoMuon/Navigation/interface/MuonLayerSort.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
/* C++ Headers */
#include <algorithm>

extern float calculateEta(float r, float z)  {
  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}

MuonEtaRange MuonNavigableLayer::TrackingRange(const FreeTrajectoryState& fts) const
{  
  float z = fts.position().z();
  float r = fts.position().perp();
  float eta;
  if ( z>0 ) eta = -log((tan(atan(r/z)/2.)));
  else eta = log(-(tan(atan(r/z)/2.)));

  float theta = atan(r/z);

  // FIXME safety factor put by hand
  float spread = 5.0*sqrt(fts.curvilinearError().matrix()(2,2))/fabs(sin(theta));  //5*sigma(eta)

  MuonEtaRange range(eta+spread,eta-spread);

  // @Chang from RB: I don't understand if the below correction is supposed only 
  // for the two other cases or not. If not this line should go before the previous.
  if ( spread < 0.07 ) spread = 0.07; 

  if ( eta > 1.0 && eta < 1.1 )  range = MuonEtaRange(eta+3.0*spread,eta-spread);
  if ( eta < -1.0 && eta > -1.1 ) range = MuonEtaRange(eta+spread,eta-3.0*spread);

  return range;
}

