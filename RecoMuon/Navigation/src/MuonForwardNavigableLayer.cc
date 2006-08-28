/** \class MuonForwardNavigableLayer
 *
 *  Navigable layer for Forward Muon
 *
 * $Date: 2006/06/04 18:39:55 $
 * $Revision: 1.5 $
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

vector<const DetLayer*> 
MuonForwardNavigableLayer::nextLayers(PropagationDirection dir) const {

  vector<const DetLayer*> result;
  vector<const DetLayer*> barrel;

  if ( dir == alongMomentum ) {
    pushResult(result, theOuterEndcapLayers);
  }
  else {
    pushResult(result, theInnerEndcapLayers);
    pushResult(barrel, theInnerBarrelLayers);
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
    pushResult(result, theInnerBarrelLayers, fts);
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
    pushResult(barrel, theAllInnerBarrelLayers);
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
    pushCompatibleResult(result, theAllInnerBarrelLayers, fts);
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

