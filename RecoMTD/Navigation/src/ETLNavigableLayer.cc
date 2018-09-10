/** \class ETLNavigableLayer
 *
 *  Navigable layer for ETL
 *
 *
 * \author : L. Gray - FNAL
 *
 *
 * Adapted from ETLNavigableLayer
 */

#include "RecoMTD/Navigation/interface/ETLNavigableLayer.h"

/* Collaborating Class Header */
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

using namespace std;
using namespace edm;

vector<const DetLayer*> 
ETLNavigableLayer::nextLayers(NavigationDirection dir) const {

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
ETLNavigableLayer::nextLayers(const FreeTrajectoryState& fts,
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
ETLNavigableLayer::compatibleLayers(NavigationDirection dir) const {

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
ETLNavigableLayer::compatibleLayers(const FreeTrajectoryState& fts,
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

void ETLNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapB& map) const {

  for (MapBI i = map.begin(); i != map.end(); i++) result.push_back((*i).first);

}


void ETLNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapE& map) const {

  for (MapEI i = map.begin(); i != map.end(); i++) result.push_back((*i).first);

}


void ETLNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapE& map,
                                           const FreeTrajectoryState& fts) const {

  for (MapEI i = map.begin(); i != map.end(); i++) 
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first);

}


void ETLNavigableLayer::pushResult(vector<const DetLayer*>& result,
                                           const MapB& map, 
                                           const FreeTrajectoryState& fts) const {

  for (MapBI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isInside(fts.position().eta())) result.push_back((*i).first);

}


void ETLNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapB& map,
                                          const FreeTrajectoryState& fts) const {
  MTDEtaRange range=trackingRange(fts);
  for ( MapBI i = map.begin(); i != map.end(); i++ )
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);
}

void ETLNavigableLayer::pushCompatibleResult(vector<const DetLayer*>& result,
                                          const MapE& map,
                                          const FreeTrajectoryState& fts) const {
  MTDEtaRange range=trackingRange(fts);
  for (MapEI i = map.begin(); i != map.end(); i++)
    if ((*i).second.isCompatible(range)) result.push_back((*i).first);

}


const DetLayer* ETLNavigableLayer::detLayer() const {

  return theDetLayer;

}


void ETLNavigableLayer::setDetLayer(const DetLayer* dl) {

  edm::LogError ("ETLNavigablaLayer") << "ETLNavigableLayer::setDetLayer called!! " << endl;

}


void ETLNavigableLayer::setInwardLinks(const MapB& innerBL,
                                               const MapE& innerEL) {

  theInnerBarrelLayers = innerBL;
  theInnerEndcapLayers = innerEL;

}
void ETLNavigableLayer::setInwardCompatibleLinks(const MapB& innerCBL,
                                               const MapE& innerCEL) {

  theAllInnerBarrelLayers = innerCBL;
  theAllInnerEndcapLayers = innerCEL;

}

