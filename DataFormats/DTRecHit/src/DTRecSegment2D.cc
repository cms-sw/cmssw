/** \file
 *
 * $Date: 2006/02/23 10:32:04 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
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
  return std::vector<TrackingRecHit*>();
}

std::vector<DTRecHit1D> DTRecSegment2D::specificRecHits() const {
  return theHits;
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
