/** \class BTLNavigableLayer
 *
 *  Navigable layer for Barrel Timing Layer
 *  Adapted from MuonBarrelNavigableLayer
 *
 *
 * \author : L. Gray
 *
 */

#include "RecoMTD/Navigation/interface/BTLNavigableLayer.h"

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
/* C++ Headers */
#include <algorithm>

using namespace std;
std::vector<const DetLayer*> 
BTLNavigableLayer::nextLayers(NavigationDirection dir) const {
  
  std::vector<const DetLayer*> result;
  
  if ( dir == insideOut ) {
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


std::vector<const DetLayer*> 
BTLNavigableLayer::nextLayers(const FreeTrajectoryState& fts,
                                     PropagationDirection dir) const {

  std::vector<const DetLayer*> result;

  if ( (isInsideOut(fts) && dir == alongMomentum) || ( !isInsideOut(fts) && dir == oppositeToMomentum)) {
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

std::vector<const DetLayer*>
BTLNavigableLayer::compatibleLayers(NavigationDirection dir) const {

  std::vector<const DetLayer*> result;

  if ( dir == insideOut ) {
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

std::vector<const DetLayer*>
BTLNavigableLayer::compatibleLayers(const FreeTrajectoryState& fts,
                                     PropagationDirection dir) const {
  std::vector<const DetLayer*> result;

  if ( (isInsideOut(fts) && dir == alongMomentum) || ( !isInsideOut(fts) && dir == oppositeToMomentum)) {
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


void BTLNavigableLayer::pushResult(std::vector<const DetLayer*>& result,
                                          const MapB& map) const {

  for ( MapBI i = map.begin(); i != map.end(); i++ ) result.push_back((*i).first); 

}

void BTLNavigableLayer::pushResult(std::vector<const DetLayer*>& result,
                                          const MapE& map) const {

  for ( MapEI i = map.begin(); i != map.end(); i++ ) result.push_back((*i).first);  
}


void BTLNavigableLayer::pushResult(std::vector<const DetLayer*>& result,
                                          const MapB& map, 
                                          const FreeTrajectoryState& fts) const {
  for ( MapBI i = map.begin(); i != map.end(); i++ ) 
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first); 
}

void BTLNavigableLayer::pushResult(std::vector<const DetLayer*>& result,
                                          const MapE& map, 
                                          const FreeTrajectoryState& fts) const {

  for (MapEI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first); 

}

void BTLNavigableLayer::pushCompatibleResult(std::vector<const DetLayer*>& result,
                                          const MapB& map,
                                          const FreeTrajectoryState& fts) const {
  MTDEtaRange range= trackingRange(fts);
  for ( MapBI i = map.begin(); i != map.end(); i++ )
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);
}

void BTLNavigableLayer::pushCompatibleResult(std::vector<const DetLayer*>& result,
                                          const MapE& map,
                                          const FreeTrajectoryState& fts) const {
  MTDEtaRange range= trackingRange(fts);
  for (MapEI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);

}

const DetLayer* BTLNavigableLayer::detLayer() const {
  return theDetLayer;
}


void BTLNavigableLayer::setDetLayer(const DetLayer* dl) {
  edm::LogError("BTLNavigableLayer") << "BTLNavigableLayer::setDetLayer called!! " << endl;
}


void BTLNavigableLayer::setInwardLinks(const MapB& innerBL) {
  theInnerBarrelLayers = innerBL;
}
void BTLNavigableLayer::setInwardCompatibleLinks(const MapB& innerCBL) {

  theAllInnerBarrelLayers = innerCBL;

}

