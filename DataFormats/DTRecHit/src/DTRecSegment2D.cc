/** \file
 *
 * $Date: 2006/03/20 12:42:29 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

/* Collaborating Class Header */

/* C++ Headers */
#include <iostream>
using namespace std;

/* ====================================================================== */

/// Constructor
DTRecSegment2D::DTRecSegment2D(const DetId& id) : theDetId(id){
}

DTRecSegment2D::DTRecSegment2D(const DetId& id, const vector<DTRecHit1D>& hits) :
theDetId(id), theChi2(0.0), theHits(hits){
}

DTRecSegment2D::DTRecSegment2D(const DetId &id, 
	       LocalPoint &position, LocalVector &direction,
	       AlgebraicSymMatrix & covMatrix, double &chi2, 
	       std::vector<DTRecHit1D> &hits1D):
  theDetId(id), thePosition(position),theDirection(direction),
  theCovMatrix(covMatrix),theChi2(chi2), theHits(hits1D){}

/// Destructor
DTRecSegment2D::~DTRecSegment2D() {
}

/* Operations */ 
LocalError DTRecSegment2D::localPositionError() const {
  return LocalError(theCovMatrix[1][1],0.,0.);
}

LocalError DTRecSegment2D::localDirectionError() const{
  return LocalError(theCovMatrix[0][0],0.,0.);
}

int DTRecSegment2D::degreesOfFreedom() const {
  return theHits.size()-dimension();
}

ostream& operator<<(ostream& os, const DTRecSegment2D& seg) {
  os << "Pos " << seg.localPosition() << 
    " Dir: " << seg.localDirection() <<
    " chi2/ndof: " << seg.chi2() << "/" << seg.degreesOfFreedom() ;
  return os;
}

std::vector<const TrackingRecHit*> DTRecSegment2D::recHits() const {
  return std::vector<const TrackingRecHit*>();
}

std::vector<TrackingRecHit*> DTRecSegment2D::recHits() {

  std::vector<TrackingRecHit*> pointersOfRecHits; 
    
  for(std::vector<DTRecHit1D>::iterator rechit = theHits.begin();
      rechit != theHits.end(); rechit++)
    pointersOfRecHits.push_back( &(*rechit) );
  
  return pointersOfRecHits;
}

std::vector<DTRecHit1D> DTRecSegment2D::specificRecHits() const {
  return theHits;
}

void DTRecSegment2D::update(std::vector<DTRecHit1D> & updatedRecHits){
  theHits = updatedRecHits;
}

void DTRecSegment2D::setPosition(const LocalPoint& pos){
  thePosition= pos;
}

void DTRecSegment2D::setDirection(const LocalVector& dir){
  theDirection=dir;
}

void DTRecSegment2D::setCovMatrix(const AlgebraicSymMatrix& cov){ 
  theCovMatrix = cov;
}

void DTRecSegment2D::setChi2(const double& chi2) {
  theChi2=chi2;
}

void DTRecSegment2D::setT0(const double& t0){
  theT0=t0;
}
