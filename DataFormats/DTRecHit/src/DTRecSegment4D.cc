/** \file
 *
 * $Date: 2009/09/21 10:13:37 $
 * $Revision: 1.15 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

/* Collaborating Class Header */
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/Utilities/interface/Exception.h"
/* C++ Headers */



DTRecSegment4D::DTRecSegment4D(const DTChamberRecSegment2D& phiSeg,
			       const DTSLRecSegment2D& zedSeg,  
			       const LocalPoint& posZInCh,
			       const LocalVector& dirZInCh):
  RecSegment(phiSeg.chamberId()), 
  theProjection(full),
  thePhiSeg(phiSeg),
  theZedSeg(zedSeg),
  theDimension(4)
{
  // Check consistency of 2 sub-segments
  if(DTChamberId(phiSeg.geographicalId().rawId()) != DTChamberId(zedSeg.geographicalId().rawId()))
    throw cms::Exception("DTRecSegment4D")
      <<"the z Segment and the phi segment have different chamber id"<<std::endl;

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

  // set cov matrix
  theCovMatrix=AlgebraicSymMatrix(4);
  theCovMatrix[0][0]=phiSeg.covMatrix()[0][0]; //sigma (dx/dz)
  theCovMatrix[0][2]=phiSeg.covMatrix()[0][1]; //cov(dx/dz,x)
  theCovMatrix[2][2]=phiSeg.covMatrix()[1][1]; //sigma (x)
  setCovMatrixForZed(posZInCh);
}


DTRecSegment4D::DTRecSegment4D(const DTChamberRecSegment2D& phiSeg) :
  RecSegment(phiSeg.chamberId()), 
  theProjection(phi),
  thePhiSeg(phiSeg),
  theZedSeg(DTSLRecSegment2D()),
  theDimension(2)
{
  thePosition=thePhiSeg.localPosition();
  
  theDirection=thePhiSeg.localDirection();

  // set cov matrix
  theCovMatrix=AlgebraicSymMatrix(4);
  theCovMatrix[0][0]=phiSeg.covMatrix()[0][0]; //sigma (dx/dz)
  theCovMatrix[0][2]=phiSeg.covMatrix()[0][1]; //cov(dx/dz,x)
  theCovMatrix[2][2]=phiSeg.covMatrix()[1][1]; //sigma (x)
}


DTRecSegment4D::DTRecSegment4D(const DTSLRecSegment2D& zedSeg,
			       const LocalPoint& posZInCh,
			       const LocalVector& dirZInCh) :
  RecSegment(zedSeg.superLayerId().chamberId()),
  theProjection(Z),
  thePhiSeg(DTChamberRecSegment2D()),
  theZedSeg(zedSeg),
  theDimension(2)
{
  
  LocalPoint posZAt0=posZInCh+
    dirZInCh*(-posZInCh.z()/cos(dirZInCh.theta()));
  
  thePosition=posZAt0;
  theDirection = dirZInCh;

  // set cov matrix
  theCovMatrix=AlgebraicSymMatrix(4);
  setCovMatrixForZed(posZInCh);
}


DTRecSegment4D::~DTRecSegment4D() {}


AlgebraicVector DTRecSegment4D::parameters() const {
  if (dimension()==4) {
    // (dx/dz,dy/dz,x,y)
    AlgebraicVector result(4);
    result[2] = thePosition.x();
    result[3] = thePosition.y();
    result[0] = theDirection.x()/theDirection.z();
    result[1] = theDirection.y()/theDirection.z();    
    return result;
  } 

  AlgebraicVector result(2);
  if (theProjection==phi) {
    // (dx/dz,x)  
    result[1] = thePosition.x();
    result[0] = theDirection.x()/theDirection.z();
  } else if (theProjection==Z) {
    // (dy/dz,y) (note we are in the chamber r.f.)
    result[1] = thePosition.y();
    result[0] = theDirection.y()/theDirection.z();
  }
  return result;
}


AlgebraicSymMatrix DTRecSegment4D::parametersError() const { 

  if (dimension()==4) {
    return theCovMatrix;
  }

  AlgebraicSymMatrix result(2);
  if (theProjection==phi) {
    result[0][0] = theCovMatrix[0][0]; //S(dx/dz)
    result[0][1] = theCovMatrix[0][2]; //Cov(dx/dz,x)
    result[1][1] = theCovMatrix[2][2]; //S(x)
  } else if (theProjection==Z) {
    result[0][0] = theCovMatrix[1][1]; //S(dy/dz)
    result[0][1] = theCovMatrix[1][3]; //Cov(dy/dz,y)
    result[1][1] = theCovMatrix[3][3]; //S(y)
  }
  return result;
}


AlgebraicMatrix DTRecSegment4D::projectionMatrix() const {
  static bool isInitialized=false;
  static AlgebraicMatrix the4DProjectionMatrix(4, 5, 0); 
  static AlgebraicMatrix the2DPhiProjMatrix(2, 5, 0);
  static AlgebraicMatrix the2DZProjMatrix(2, 5, 0);

  if (!isInitialized) {
    the4DProjectionMatrix[0][1] = 1;
    the4DProjectionMatrix[1][2] = 1;
    the4DProjectionMatrix[2][3] = 1;
    the4DProjectionMatrix[3][4] = 1;

    the2DPhiProjMatrix[0][1] = 1;
    the2DPhiProjMatrix[1][3] = 1;

    the2DZProjMatrix[0][2] = 1;
    the2DZProjMatrix[1][4] = 1;

    isInitialized= true;
  }

  if (dimension()==4) { 
    return the4DProjectionMatrix;
  } else if (theProjection==phi) {
    return the2DPhiProjMatrix;
  } else if (theProjection==Z) {
    return the2DZProjMatrix;
  } else {
    return AlgebraicMatrix();
  }
}


LocalError DTRecSegment4D::localPositionError() const {
  return LocalError(theCovMatrix[2][2],theCovMatrix[2][3],theCovMatrix[3][3]);
}


LocalError DTRecSegment4D::localDirectionError() const {
  return LocalError(theCovMatrix[0][0],theCovMatrix[0][1],theCovMatrix[1][1]);
}


double DTRecSegment4D::chi2() const {
  double result=0;
  if (hasPhi()) result+=thePhiSeg.chi2();
  if (hasZed()) result+=theZedSeg.chi2();
  return result;
}


int DTRecSegment4D::degreesOfFreedom() const {
  int result=0;
  if (hasPhi()) result+=thePhiSeg.degreesOfFreedom();
  if (hasZed()) result+=theZedSeg.degreesOfFreedom();
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
    2.*(posZInCh.z()*posZInCh.z())*theZedSeg.parametersError()[0][0] +
    theZedSeg.parametersError()[1][1] +
    2.*posZInCh.z()*theZedSeg.parametersError()[0][1];
  // cout << " = " << theCovMatrix[3][3] << endl;
}

std::ostream& operator<<(std::ostream& os, const DTRecSegment4D& seg) {
  os << "Pos " << seg.localPosition() << 
    " Dir: " << seg.localDirection() <<
    " dim: " << seg.dimension() <<
    " chi2/ndof: " << seg.chi2() << "/" << seg.degreesOfFreedom() << " :";
  if (seg.hasPhi()) os << seg.phiSegment()->recHits().size();
  else os << 0;
  os << ":";
  if (seg.hasZed()) os << seg.zSegment()->recHits().size();
  else os << 0;
  return os;
}


/// Access to component RecHits (if any)
std::vector<const TrackingRecHit*> DTRecSegment4D::recHits() const{
  std::vector<const TrackingRecHit*> pointersOfRecHits; 

  if (hasPhi()) pointersOfRecHits.push_back(phiSegment());
  if (hasZed()) pointersOfRecHits.push_back(zSegment());

  return pointersOfRecHits;
}


/// Non-const access to component RecHits (if any)
std::vector<TrackingRecHit*> DTRecSegment4D::recHits(){

  std::vector<TrackingRecHit*> pointersOfRecHits; 

  if (hasPhi()) pointersOfRecHits.push_back(phiSegment());
  if (hasZed()) pointersOfRecHits.push_back(zSegment());
  
  return pointersOfRecHits;
}


DTChamberId DTRecSegment4D::chamberId() const {
  return DTChamberId(geographicalId());
}
