#include "RecoMuon/Navigation/interface/MuonNavigableLayer.h"
//   Ported from ORCA.
//   New methods compatibleLayers are added.
//   $Date: 2006/03/22 02:10:14 $
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
/// Estimate an eta range for FTS
MuonEtaRange MuonNavigableLayer::TrackingRange(const FreeTrajectoryState& fts) const
{  
  float z = fts.position().z();
  float r = fts.position().perp();
  float eta= log(-(tan(atan(r/z)/2.)));
  if ( z>0 ) eta=-log((tan(atan(r/z)/2.)));
  float theta2 = atan(r/z)/2.;
  float spread = 5.0*sqrt(fts.curvilinearError().matrix()(2,2))/(2.0*sin(theta2)*cos(theta2));  //5*sigma(eta)
  float eta_max=0;
  if ( z > 0 ) eta_max = -log((tan(atan(r/(z+spread))/2.)));
  else eta_max = log(-(tan(atan(r/(z-spread))/2.)));

  spread = fabs(eta_max-eta);
  MuonEtaRange range(eta+spread,eta-spread);

  if ( spread < 0.07 ) spread = 0.07;
  if ( eta > 1.0 && eta < 1.1 ) MuonEtaRange range(eta+3.0*spread,eta-spread);
  if ( eta < -1.0 && eta > -1.1 ) MuonEtaRange range(eta+spread,eta-3.0*spread);
  return range;
}

