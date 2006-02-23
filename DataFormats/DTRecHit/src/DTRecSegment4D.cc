/** \file
 *
 * $Date:  22/02/2006 15:20:44 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

/* Collaborating Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment2DPhi.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/// Constructor
DTRecSegment4D::DTRecSegment4D(DTRecSegment2DPhi* phiSeg, DTRecSegment2D* zedSeg) :
thePhiSeg(phiSeg), theZedSeg(zedSeg){
}

DTRecSegment4D::DTRecSegment4D(DTRecSegment2DPhi* phiSeg) :
thePhiSeg(phiSeg), theZedSeg(0){
}

DTRecSegment4D::DTRecSegment4D(DTRecSegment2D* zedSeg) :
thePhiSeg(0), theZedSeg(zedSeg){
}

/// Destructor
DTRecSegment4D::~DTRecSegment4D() {
  delete thePhiSeg;
  delete theZedSeg;
}

/* Operations */ 
AlgebraicVector DTRecSegment4D::parameters() const {
  AlgebraicVector result(2);
  if (dimension()==4) return DTRecSegment4D::parameters();
  else {
    if (thePhiSeg) {
      result[1] = localPosition().x();
      result[0] = localDirection().x()/localDirection().z();
    } else {
      result[1] = localPosition().y();
      result[0] = localDirection().y()/localDirection().z();
    }
  }
  return result;
}


AlgebraicSymMatrix DTRecSegment4D::parametersError() const { 
  AlgebraicSymMatrix result(2);
  if (dimension()==4) return theCovMatrix;
  else {
    if (thePhiSeg) {
      result[0][0] = theCovMatrix[0][0]; //S(dx/dz)
      result[0][1] = theCovMatrix[0][2]; //Cov(dx/dz,x)
      result[1][1] = theCovMatrix[2][2]; //S(x)
    } else {
      result[0][0] = theCovMatrix[1][1]; //S(dy/dz)
      result[0][1] = theCovMatrix[1][3]; //Cov(dy/dz,y)
      result[1][1] = theCovMatrix[3][3]; //S(y)
    }
  }
  return result;
}

LocalError DTRecSegment4D::localPositionError() const {
  return LocalError(theCovMatrix[2][2],theCovMatrix[2][3],theCovMatrix[3][3]);
}

LocalError DTRecSegment4D::localDirectionError() const {
  return LocalError(theCovMatrix[0][0],theCovMatrix[0][1],theCovMatrix[1][1]);
}

double DTRecSegment4D::chi2() const {
  double result=0;
  if (thePhiSeg) result+=thePhiSeg->chi2();
  if (theZedSeg) result+=theZedSeg->chi2();
  return result;
}

int DTRecSegment4D::degreesOfFreedom() const {
  int result=0;
  if (thePhiSeg) result+=thePhiSeg->degreesOfFreedom();
  if (theZedSeg) result+=theZedSeg->degreesOfFreedom();
  return result;
}

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg) {
  os << "Pos " << seg.localPosition() << 
    " Dir: " << seg.localDirection() <<
    " dim: " << seg.dimension() <<
    " chi2/ndof: " << seg.chi2() << "/" << seg.degreesOfFreedom() ;
  return os;
}
