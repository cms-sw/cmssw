/** \file
 *
 * $Date: 2006/04/19 15:08:06 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

/* Collaborating Class Header */
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

#include "FWCore/Utilities/interface/Exception.h"
/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */
// FIXME
// is the choice .specificRecHits().size() the best way to do the check??


/// Constructor
DTRecSegment4D::DTRecSegment4D(const DTRecSegment2DPhi& phiSeg, const DTRecSegment2D& zedSeg, const DTChamber* chamber):
  thePhiSeg(phiSeg),theZedSeg(zedSeg){
  
  if(zedSeg.specificRecHits().size()){ 
    if(DTChamberId(phiSeg.geographicalId().rawId()) != DTChamberId(zedSeg.geographicalId().rawId()))
      throw cms::Exception("DTRecSegment4D")
	<<"the z Segment and the phi segment have different chamber id"<<std::endl;
  }
  
  theDetId = DTChamberId(phiSeg.geographicalId().rawId());

  // FIXME: do the porting of this c'tor!!!!
}

DTRecSegment4D::DTRecSegment4D(const DTRecSegment2DPhi& phiSeg, const DTChamber* chamber) :
  thePhiSeg(phiSeg), theZedSeg( DTRecSegment2D() ){
  theDetId = DTChamberId(phiSeg.geographicalId().rawId());

  // FIXME: do the porting of this c'tor!!!!
}

DTRecSegment4D::DTRecSegment4D(const DTRecSegment2D& zedSeg, const DTChamber* chamber) :
thePhiSeg( DTRecSegment2DPhi() ), theZedSeg( zedSeg){
  theDetId = DTChamberId(zedSeg.geographicalId().rawId());

  // FIXME: do the porting of this c'tor!!!!
}

/// Destructor
DTRecSegment4D::~DTRecSegment4D() {}

/* Operations */ 
AlgebraicVector DTRecSegment4D::parameters() const {
  AlgebraicVector result(2);
  if (dimension()==4) return DTRecSegment4D::parameters();
  else {
    if (thePhiSeg.specificRecHits().size()) {
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
    if (thePhiSeg.specificRecHits().size()) {
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
  if (thePhiSeg.specificRecHits().size()) result+=thePhiSeg.chi2();
  if (theZedSeg.specificRecHits().size()) result+=theZedSeg.chi2();
  return result;
}

int DTRecSegment4D::degreesOfFreedom() const {
  int result=0;
  if (thePhiSeg.specificRecHits().size()) result+=thePhiSeg.degreesOfFreedom();
  if (theZedSeg.specificRecHits().size()) result+=theZedSeg.degreesOfFreedom();
  return result;
}

void DTRecSegment4D::setCovMatrixForZed(const LocalPoint& posZInCh){
  // Warning!!! the covariance matrix for Theta SL segment is defined in the SL
  // reference frame, here that in the Chamber ref frame must be used.
  // For direction, no problem, but the position is extrapolated, so we must
  // propagate the error properly.

  // many thanks to Paolo Ronchese for the help in deriving the formulas!

  // y=m*z+q in SL frame
  // y=m'*z+q' in CH frame

  // var(m') = var(m)
  theCovMatrix[1][1] = theZedSeg.parametersError()[0][0]; //sigma (dy/dz)

  // cov(m',q') = DeltaZ*Var(m) + Cov(m,q)
  theCovMatrix[1][3] =
    posZInCh.z()*theZedSeg.parametersError()[0][0]+
    theZedSeg.parametersError()[0][1]; //cov(dy/dz,y)

  // Var(q') = DeltaZ^2*Var(m) + Var(q) + 2*DeltaZ*Cov(m,q)
  // cout << "Var(q') = DeltaZ^2*Var(m) + Var(q) + 2*DeltaZ*Cov(m,q)" << endl;
  // cout << "Var(q')= " << posZInCh.z()*posZInCh.z() << "*" <<
  //   theZedSeg.parametersError()[0][0] << " + " << 
  //   theZedSeg.parametersError()[1][1] << " + " << 
  //   2*posZInCh.z() << "*" << theZedSeg.parametersError()[0][1] ;
  theCovMatrix[3][3] =
    (posZInCh.z()*posZInCh.z())*theZedSeg.parametersError()[0][0] +
    theZedSeg.parametersError()[1][1] +
    2*posZInCh.z()*theZedSeg.parametersError()[0][1];
  // cout << " = " << theCovMatrix[3][3] << endl;
}

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg) {
  os << "Pos " << seg.localPosition() << 
    " Dir: " << seg.localDirection() <<
    " dim: " << seg.dimension() <<
    " chi2/ndof: " << seg.chi2() << "/" << seg.degreesOfFreedom() ;
  return os;
}


// DTChamberId DTRecSegment4D::chamberId() const{
//   if(phiSegment()->chamberId() == zSegment()->superLayerId().chamberId())
//     return phiSegment()->chamberId();
//   else 
//     throw cms::Exception("DTRecSegment4D")
//       <<"the z Segment and the phi segment have different chamber id"<<std::endl;
// }


/// Access to component RecHits (if any)
std::vector<const TrackingRecHit*> DTRecSegment4D::recHits() const{
  return std::vector<const TrackingRecHit*>();
}

/// Non-const access to component RecHits (if any)
std::vector<TrackingRecHit*> DTRecSegment4D::recHits(){
  return std::vector<TrackingRecHit*>(); 
}
