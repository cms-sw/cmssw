/** \class MTDNavigableLayer
 *
 *  base class for MTDNavigableLayer and MuonForwardNavigableLayer.
 *  trackingRange defines an MTDEtaRange for an FTS, 
 *  which is used for searching compatible DetLayers.
 *
 *
 * \author : Chang Liu - Purdue University <Chang.Liu@cern.ch>
 * with contributions from: R. Bellan - INFN Torino
 *
 * code of trackingRange is from MuonGlobalNavigation in ORCA
 * whose author is Stefano Lacaprara - INFN Padova
 * Modification:
 *
 */

#include "RecoMTD/Navigation/interface/MTDNavigableLayer.h"

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
/* C++ Headers */
#include <algorithm>


using namespace std;

extern float calculateEta(float r, float z)  {
  if ( z > 0 ) return -log((tan(atan(r/z)/2.)));
  return log(-(tan(atan(r/z)/2.)));

}

MTDEtaRange MTDNavigableLayer::trackingRange(const FreeTrajectoryState& fts) const
{  
  float z = fts.position().z();
  float r = fts.position().perp();
  float eta;
  if ( z>0 ) eta = -log((tan(atan(r/z)/2.)));
  else eta = log(-(tan(atan(r/z)/2.)));

  double theta = atan(r/z);

  double spread = 5.0*sqrt(fts.curvilinearError().matrix()(2,2))/fabs(sin(theta));  //5*sigma(eta)

  //C.L.: this spread could be too large to use.
  // convert it to a smaller one by assuming a virtual radius
  // that transforms the error on angle to error on z axis.
  // not accurate, but works!

  double eta_max = 0;

  if ( z > 0 ) eta_max = calculateEta(r, z+spread); 
  else eta_max = calculateEta(r, z-spread); 

  spread = std::min(0.07, fabs(eta_max-eta));

  MTDEtaRange range(eta+spread,eta-spread);

  spread = 0.07; 
  // special treatment for special geometry in overlap region
  
  if ( eta > 1.0 && eta < 1.1 )  range = MTDEtaRange(eta+3.0*spread,eta-spread);
  if ( eta < -1.0 && eta > -1.1 ) range = MTDEtaRange(eta+spread,eta-3.0*spread);

  return range;
}

bool MTDNavigableLayer::isInsideOut(const FreeTrajectoryState& fts) const {
  
  return (fts.position().basicVector().dot(fts.momentum().basicVector())>0);
  
}

