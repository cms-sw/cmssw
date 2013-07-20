/** \file
 *
 * $Date: 2012/04/30 08:32:40 $
 * $Revision: 1.12 $
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


/* static member definition */
bool DTRecSegment2D::isInitialized(false);

AlgebraicMatrix DTRecSegment2D::theProjectionMatrix;

/* Operations */ 
AlgebraicSymMatrix DTRecSegment2D::parametersError() const {
  AlgebraicSymMatrix m(2);
  /// mat[0][0]=sigma (dx/dz)
  /// mat[1][1]=sigma (x)
  /// mat[0][1]=cov(dx/dz,x)
  // if ( det().alignmentPositionError()) {
  //   LocalError lape = 
  //     ErrorFrameTransformer().transform( det().alignmentPositionError()->globalError(), 
  //                                        det().surface());
  //   m[0][0] = lv.xx();
  //   m[0][1] = 0.;
  //   m[1][1] = lp.xx()+lape.xx();
  // } else {
    m[0][0] = theCovMatrix[0][0];
    m[0][1] = theCovMatrix[0][1];
    m[1][1] = theCovMatrix[1][1];
  //};

    //cout << "theCovMatrix elements " << theCovMatrix[0][0] << " , " << theCovMatrix[0][1] <<
    //        " , " << theCovMatrix[1][0] << " , " << theCovMatrix[1][1] << endl;

  return m;

}

DTRecSegment2D::~DTRecSegment2D(){}

DTRecSegment2D::DTRecSegment2D(DetId id, const vector<DTRecHit1D>& hits) :
  RecSegment(id), theChi2(0.0), theT0(0.), theVdrift(0.), theHits(hits){
}

DTRecSegment2D::DTRecSegment2D(DetId id, 
	       LocalPoint &position, LocalVector &direction,
	       AlgebraicSymMatrix & covMatrix, double chi2, 
	       std::vector<DTRecHit1D> &hits1D):
 RecSegment(id), thePosition(position),theDirection(direction),
  theCovMatrix(covMatrix),theChi2(chi2),theT0(0.),theVdrift(0.),theHits(hits1D){}

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

  std::vector<const TrackingRecHit*> pointersOfRecHits; 
  
  for(std::vector<DTRecHit1D>::const_iterator rechit = theHits.begin();
      rechit != theHits.end(); rechit++)
    pointersOfRecHits.push_back( &(*rechit) );
  
  return pointersOfRecHits;
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

void DTRecSegment2D::setVdrift(const double& vdrift){
  theVdrift=vdrift;
}
