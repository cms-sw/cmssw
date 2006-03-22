#include "RecoMuon/Navigation/interface/MuonBarrelNavigableLayer.h"
//   Ported from ORCA.
//   New methods compatibleLayers are added.
//   $Date: $
//   $Revision: $

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
vector<const DetLayer*> 
MuonBarrelNavigableLayer::nextLayers(PropagationDirection dir) const {

  vector<const DetLayer*> result;

  if ( dir == alongMomentum ) {
    pushResult(result, theOuterBarrelLayers);
    pushResult(result, theOuterBackwardLayers);
    pushResult(result, theOuterForwardLayers);
  }
  else {
    pushResult(result, theInnerBarrelLayers);
    reverse(result.begin(),result.end());
    pushResult(result, theInnerBackwardLayers);
    pushResult(result, theInnerForwardLayers);
  }

  result.reserve(result.size());
  return result;

}


vector<const DetLayer*> 
MuonBarrelNavigableLayer::nextLayers(const FreeTrajectoryState& fts,
                                     PropagationDirection dir) const {

  vector<const DetLayer*> result;

  if ( dir == alongMomentum ) {
    pushResult(result, theOuterBarrelLayers, fts);
    pushResult(result, theOuterBackwardLayers, fts);
    pushResult(result, theOuterForwardLayers, fts);
  }
  else {
    pushResult(result, theInnerBarrelLayers, fts);
    reverse(result.begin(),result.end());
    pushResult(result, theInnerBackwardLayers, fts);
    pushResult(result, theInnerForwardLayers, fts);
  }
  result.reserve(result.size());
  return result;
}

vector<const DetLayer*>
MuonBarrelNavigableLayer::compatibleLayers(PropagationDirection dir) const {

  vector<const DetLayer*> result;

  if ( dir == alongMomentum ) {
    pushResult(result, theAllOuterBarrelLayers);
    pushResult(result, theAllOuterBackwardLayers);
    pushResult(result, theAllOuterForwardLayers);
  }
  else {
    pushResult(result, theAllInnerBarrelLayers);
    reverse(result.begin(),result.end());
    pushResult(result, theAllInnerBackwardLayers);
    pushResult(result, theAllInnerForwardLayers);
  }

  result.reserve(result.size());
  return result;
}

vector<const DetLayer*>
MuonBarrelNavigableLayer::compatibleLayers(const FreeTrajectoryState& fts,
                                     PropagationDirection dir) const {
  vector<const DetLayer*> result;
  if ( dir == alongMomentum ) {
    pushCompatibleResult(result, theAllOuterBarrelLayers, fts);
    pushCompatibleResult(result, theAllOuterBackwardLayers, fts);
    pushCompatibleResult(result, theAllOuterForwardLayers, fts);
  }
  else {
    pushCompatibleResult(result, theAllInnerBarrelLayers, fts);
    reverse(result.begin(),result.end());
    pushCompatibleResult(result, theAllInnerBackwardLayers, fts);
    pushCompatibleResult(result, theAllInnerForwardLayers, fts);
  }
  result.reserve(result.size());
  return result;

}


void MuonBarrelNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                          const MapB& map) const {

  for ( MapBI i = map.begin(); i != map.end(); i++ ) result.push_back((*i).first); 

}

void MuonBarrelNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                          const MapE& map) const {

  for ( MapEI i = map.begin(); i != map.end(); i++ ) result.push_back((*i).first);  
}


void MuonBarrelNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                          const MapB& map, 
                                          const FreeTrajectoryState& fts) const {
  for ( MapBI i = map.begin(); i != map.end(); i++ ) 
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first); 
}

void MuonBarrelNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                          const MapE& map, 
                                          const FreeTrajectoryState& fts) const {

  for (MapEI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first); 

}

void MuonBarrelNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapB& map,
                                          const FreeTrajectoryState& fts) const {
  MuonEtaRange range= TrackingRange(fts);
  for ( MapBI i = map.begin(); i != map.end(); i++ )
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);
}

void MuonBarrelNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapE& map,
                                          const FreeTrajectoryState& fts) const {
  MuonEtaRange range= TrackingRange(fts);
  for (MapEI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);

}
/// Estimate an eta range for FTS
MuonEtaRange MuonBarrelNavigableLayer::TrackingRange(const FreeTrajectoryState& fts) const
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
 

DetLayer* MuonBarrelNavigableLayer::detLayer() const {
  return theDetLayer;
}


void MuonBarrelNavigableLayer::setDetLayer(DetLayer* dl) {
  edm::LogError("MuonBarrelNavigableLayer") << "MuonBarrelNavigableLayer::setDetLayer called!! " << endl;
}


void MuonBarrelNavigableLayer::setInwardLinks(const MapB& innerBL) {
  theInnerBarrelLayers = innerBL;
}
void MuonBarrelNavigableLayer::setInwardCompatibleLinks(const MapB& innerCBL) {

  theAllInnerBarrelLayers = innerCBL;

}

