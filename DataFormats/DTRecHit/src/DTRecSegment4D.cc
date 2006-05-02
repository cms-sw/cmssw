/** \file
 *
 * $Date: 2006/04/20 17:10:32 $
 * $Revision: 1.4 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

/* Collaborating Class Header */
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/Utilities/interface/Exception.h"
/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */
// FIXME
// is the choice .specificRecHits().size() the best way to do the check??


/// Constructor
DTRecSegment4D::DTRecSegment4D(const DTChamberRecSegment2D& phiSeg, const DTSLRecSegment2D& zedSeg,  
			       const LocalPoint& posZInCh, const LocalVector& dirZInCh):
  thePhiSeg(phiSeg),theZedSeg(zedSeg){
  
  if(zedSeg.specificRecHits().size()){ 
    if(DTChamberId(phiSeg.geographicalId().rawId()) != DTChamberId(zedSeg.geographicalId().rawId()))
      throw cms::Exception("DTRecSegment4D")
	<<"the z Segment and the phi segment have different chamber id"<<std::endl;
  }
  
  theDetId = phiSeg.chamberId();

  // The position of 2D segments are defined in the SL frame: I must first
  // extrapolate that position at the Chamber reference plane

  LocalPoint posZAt0 = posZInCh +
    dirZInCh * (-posZInCh.z())/cos(dirZInCh.theta());


  thePosition=LocalPoint(phiSeg.localPosition().x(),posZAt0.y(),0.);
  LocalVector dirPhiInCh=phiSeg.localDirection();

  // given the actual definition of chamber refFrame, (with z poiniting to IP),
  // the zed component of direction is negative.
  theDirection=LocalVector(dirPhiInCh.x()/fabs(dirPhiInCh.z()),
                           dirZInCh.y()/fabs(dirZInCh.z()),
                           -1.);
  theDirection=theDirection.unit();

  theCovMatrix=AlgebraicSymMatrix(4);

  // set cov matrix
  theCovMatrix[0][0]=phiSeg.covMatrix()[0][0]; //sigma (dx/dz)
  theCovMatrix[0][2]=phiSeg.covMatrix()[0][1]; //cov(dx/dz,x)
  theCovMatrix[2][2]=phiSeg.covMatrix()[1][1]; //sigma (x)
  
  setCovMatrixForZed(posZInCh);

  // set the projection matrix and the dimension
  theDimension=4;
  theProjMatrix = RecSegment4D::projectionMatrix();
}

DTRecSegment4D::DTRecSegment4D(const DTChamberRecSegment2D& phiSeg) :
  thePhiSeg(phiSeg), theZedSeg( DTSLRecSegment2D() ){
  
  theDetId = phiSeg.chamberId();

  thePosition=thePhiSeg.localPosition();
  
  theDirection=thePhiSeg.localDirection();

  theCovMatrix=AlgebraicSymMatrix(4);
  // set cov matrix
  theCovMatrix[0][0]=phiSeg.covMatrix()[0][0]; //sigma (dx/dz)
  theCovMatrix[0][2]=phiSeg.covMatrix()[0][1]; //cov(dx/dz,x)
  theCovMatrix[2][2]=phiSeg.covMatrix()[1][1]; //sigma (x)

  // set the projection matrix and the dimension
  theDimension=2;
  theProjMatrix=AlgebraicMatrix(2,5,0);
  theProjMatrix[0][1] = 1;
  theProjMatrix[1][3] = 1;
}

DTRecSegment4D::DTRecSegment4D(const DTSLRecSegment2D& zedSeg,
			       const LocalPoint& posZInCh, const LocalVector& dirZInCh):
  thePhiSeg( DTChamberRecSegment2D() ), theZedSeg( zedSeg){
  theDetId = zedSeg.superLayerId().chamberId();
  
  LocalPoint posZAt0=posZInCh+
    dirZInCh*(-posZInCh.z()/cos(dirZInCh.theta()));
  
  thePosition=posZAt0;
  theDirection = dirZInCh;

  theCovMatrix=AlgebraicSymMatrix(4);
  // set cov matrix
  setCovMatrixForZed(posZInCh);

  // set the projection matrix and the dimension
  theDimension=2;
  theProjMatrix=AlgebraicMatrix(2,5,0);
  theProjMatrix[0][2] = 1;
  theProjMatrix[1][4] = 1;
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
