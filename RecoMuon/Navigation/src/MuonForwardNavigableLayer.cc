/** \class MuonForwardNavigableLayer
 *
 *  Navigable layer for Forward Muon
 *
 * $Date: 2007/01/29 16:24:52 $
 * $Revision: 1.10 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 * Chang Liu:
 *  compatibleLayers(dir) and compatibleLayers(fts, dir) are added,
 *  which return ALL DetLayers that are compatible with a given DetLayer.
 */

#include "RecoMuon/Navigation/interface/MuonForwardNavigableLayer.h"

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;
using namespace edm;

vector<const DetLayer*> 
MuonForwardNavigableLayer::nextLayers(NavigationDirection dir) const {

  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;

  if ( dir == insideOut ) {
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

  if ( (isInsideOut(fts) && dir == alongMomentum) || ( !isInsideOut(fts) && dir == oppositeToMomentum)) {
    pushResult(result, theOuterEndcapLayers, fts);
  }
  else {
    pushResult(result, theInnerEndcapLayers, fts);
    reverse(result.begin(),result.end());
    pushResult(barrel, theInnerBarrelLayers, fts);
    reverse(barrel.begin(),barrel.end());
    result.insert(result.end(),barrel.begin(),barrel.end());
  }

  result.reserve(result.size());
  return result;

}

vector<const DetLayer*>
MuonForwardNavigableLayer::compatibleLayers(NavigationDirection dir) const {

  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;

  if ( dir == insideOut ) {
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
  
  if ( (isInsideOut(fts) && dir == alongMomentum) || ( !isInsideOut(fts) && dir == oppositeToMomentum)) {
    pushCompatibleResult(result, theAllOuterEndcapLayers, fts);
  }
  else {
    pushCompatibleResult(result, theAllInnerEndcapLayers, fts);
    reverse(result.begin(),result.end());
    pushCompatibleResult(barrel, theAllInnerBarrelLayers, fts);
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
  MuonEtaRange range=trackingRange(fts);
  for ( MapBI i = map.begin(); i != map.end(); i++ )
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);
}

void MuonForwardNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapE& map,
                                          const FreeTrajectoryState& fts) const {
  MuonEtaRange range=trackingRange(fts);
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

