#include "RecoMuon/Navigation/interface/MuonForwardNavigableLayer.h"

//   Ported from ORCA.
//   Two new methods compatibleLayers are added.
//   $Date: $
//   $Revision: $

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

vector<const DetLayer*> 
MuonForwardNavigableLayer::nextLayers(PropagationDirection dir) const {

  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;

  if ( dir == alongMomentum ) {
    pushResult(result, theOuterEndcapLayers);
  }
  else {
    pushResult(result, theInnerEndcapLayers);
    reverse(result.begin(),result.end());
    pushResult(barrel, theInnerBarrelLayers);
    reverse(barrel.begin(),barrel.end());
    result.insert(result.end(),barrel.begin(),barrel.end());
  }

  result.reserve(result.size());
  return result;

}


vector<const DetLayer*> 
MuonForwardNavigableLayer::nextLayers(const FreeTrajectoryState& fts,
                                      PropagationDirection dir) const {

  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;

  if ( dir == alongMomentum ) {
    pushResult(result, theOuterEndcapLayers, fts);
  }
  else {
    pushResult(result, theInnerEndcapLayers, fts);
    reverse(result.begin(),result.end());
    pushResult(result, theInnerBarrelLayers, fts);
    reverse(barrel.begin(),barrel.end());
    result.insert(result.end(),barrel.begin(),barrel.end());
  }

  result.reserve(result.size());
  return result;

}

vector<const DetLayer*>
MuonForwardNavigableLayer::compatibleLayers(PropagationDirection dir) const {

  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;

  if ( dir == alongMomentum ) {
    pushResult(result, theAllOuterEndcapLayers);
  }
  else {
    pushResult(result, theAllInnerEndcapLayers);
    reverse(result.begin(),result.end());
    pushResult(barrel, theAllInnerBarrelLayers);
    reverse(barrel.begin(),barrel.end());
    result.insert(result.end(),barrel.begin(),barrel.end());
  }

  result.reserve(result.size());
  return result;

}
vector<const DetLayer*>
MuonForwardNavigableLayer::compatibleLayers(const FreeTrajectoryState& fts,
                                      PropagationDirection dir) const {
  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;
  if ( dir == alongMomentum ) {
    pushCompatibleResult(result, theAllOuterEndcapLayers, fts);
  }
  else {
    pushCompatibleResult(result, theAllInnerEndcapLayers, fts);
    reverse(result.begin(),result.end());
    pushCompatibleResult(result, theAllInnerBarrelLayers, fts);
    reverse(barrel.begin(),barrel.end());
    result.insert(result.end(),barrel.begin(),barrel.end());
  }
  result.reserve(result.size());
  return result;

}

void MuonForwardNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapB& map) const {

  for (MapBI i = map.begin(); i != map.end(); i++) result.push_back((*i).first);

}


void MuonForwardNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapE& map) const {

  for (MapEI i = map.begin(); i != map.end(); i++) result.push_back((*i).first);

}


void MuonForwardNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapE& map,
                                           const FreeTrajectoryState& fts) const {

  for (MapEI i = map.begin(); i != map.end(); i++) 
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first);

}


void MuonForwardNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapB& map, 
                                           const FreeTrajectoryState& fts) const {

  for (MapBI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first);

}


void MuonForwardNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapB& map,
                                          const FreeTrajectoryState& fts) const {
  MuonEtaRange range=TrackingRange(fts);
  for ( MapBI i = map.begin(); i != map.end(); i++ )
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);
}

void MuonForwardNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapE& map,
                                          const FreeTrajectoryState& fts) const {
  MuonEtaRange range=TrackingRange(fts);
  for (MapEI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);

}
// Estimate an eta range for a FTS
MuonEtaRange MuonForwardNavigableLayer::TrackingRange(const FreeTrajectoryState& fts) const
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


DetLayer* MuonForwardNavigableLayer::detLayer() const {

  return theDetLayer;

}


void MuonForwardNavigableLayer::setDetLayer(DetLayer* dl) {

  edm::LogError ("MuonForwardNavigablaLayer") << "MuonForwardNavigableLayer::setDetLayer called!! " << endl;

}


void MuonForwardNavigableLayer::setInwardLinks(const MapB& innerBL,
                                               const MapE& innerEL) {

  theInnerBarrelLayers = innerBL;
  theInnerEndcapLayers = innerEL;

}
void MuonForwardNavigableLayer::setInwardCompatibleLinks(const MapB& innerCBL,
                                               const MapE& innerCEL) {

  theAllInnerBarrelLayers = innerCBL;
  theAllInnerEndcapLayers = innerCEL;

}

