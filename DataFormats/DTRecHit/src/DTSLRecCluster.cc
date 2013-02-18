/******* \class DTSLRecCluster *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 17/04/2008 17:14:32 CEST $
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/* static member definition */
static AlgebraicMatrix initMatrix()  {
  AlgebraicMatrix m( 2, 5, 0);
  m[0][1]=1; 
  return m;
}

const AlgebraicMatrix DTSLRecCluster::theProjectionMatrix = initMatrix();

/* Constructor */ 
DTSLRecCluster::DTSLRecCluster(const DTSuperLayerId id, const std::vector<DTRecHit1DPair>& pairs) :
theSlid(id), thePairs(pairs){
}

DTSLRecCluster::DTSLRecCluster(const DTSuperLayerId id,
                               const LocalPoint& pos,
                               const LocalError& err,
                               const std::vector<DTRecHit1DPair>& pairs) :
theSlid(id), thePos(pos), thePosError(err), thePairs(pairs){
}

/* Destructor */ 

/* Operations */ 
vector<const TrackingRecHit*> DTSLRecCluster::recHits() const {
  std::vector<const TrackingRecHit*> pointersOfRecHits; 
  
  for(std::vector<DTRecHit1DPair>::const_iterator rechit = thePairs.begin();
      rechit != thePairs.end(); rechit++)
    pointersOfRecHits.push_back( &(*rechit) );
  
  return pointersOfRecHits;
}

vector<TrackingRecHit*> DTSLRecCluster::recHits() {
  std::vector<TrackingRecHit*> pointersOfRecHits; 
  
  for(std::vector<DTRecHit1DPair>::iterator rechit = thePairs.begin();
      rechit != thePairs.end(); rechit++)
    pointersOfRecHits.push_back( &(*rechit) );
  
  return pointersOfRecHits;
}

ostream& operator<<(ostream& os, const DTSLRecCluster& clus) {
  os << "Pos " << clus.localPosition()
    << " err " << clus.localPositionError()
    << " nHits: " << clus.nHits() ;
  return os;
}


