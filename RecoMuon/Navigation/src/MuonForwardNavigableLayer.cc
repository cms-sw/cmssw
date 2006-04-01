#include "RecoMuon/Navigation/interface/MuonForwardNavigableLayer.h"

//   Ported from ORCA.
//   Two new methods compatibleLayers are added.
//   $Date: 2006/03/22 02:14:21 $
//   $Revision: 1.1 $

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

