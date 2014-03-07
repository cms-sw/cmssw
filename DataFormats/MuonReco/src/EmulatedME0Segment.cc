/** \file ME0Segment.cc
 *
 *  $Date: 2013/04/22 22:41:33 $
 *  \author David Nash
 */

#include <DataFormats/MuonReco/interface/EmulatedME0Segment.h>
#include <iostream>

EmulatedME0Segment::EmulatedME0Segment(LocalPoint origin,  	LocalVector direction, AlgebraicSymMatrix errors, double chi2) : 
  theOrigin(origin), 
  theLocalDirection(direction), theCovMatrix(errors), theChi2(chi2) {
}

EmulatedME0Segment::~EmulatedME0Segment() {}

LocalError EmulatedME0Segment::localPositionError() const {
  return LocalError(theCovMatrix[2][2], theCovMatrix[2][3], theCovMatrix[3][3]);
}

LocalError EmulatedME0Segment::localDirectionError() const {
  return LocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]); 
}


AlgebraicVector EmulatedME0Segment::parameters() const {
  // For consistency with DT and what we require for the TrackingRecHit interface,
  // the order of the parameters in the returned vector should be (dx/dz, dy/dz, x, z)
  
  AlgebraicVector result(4);

  result[0] = theLocalDirection.x()/theLocalDirection.z();
  result[1] = theLocalDirection.y()/theLocalDirection.z();    
  result[2] = theOrigin.x();
  result[3] = theOrigin.y();

  return result;
}


AlgebraicMatrix EmulatedME0Segment::projectionMatrix() const {
  static AlgebraicMatrix theProjectionMatrix( 4, 5, 0);
  static bool isInitialized = false;
  if (!isInitialized) {
    theProjectionMatrix[0][1] = 1;
    theProjectionMatrix[1][2] = 1;
    theProjectionMatrix[2][3] = 1;
    theProjectionMatrix[3][4] = 1;
    isInitialized=true;
  }    
  return theProjectionMatrix;
}

//
void EmulatedME0Segment::print() const {
  std::cout << *this << std::endl;
}

std::ostream& operator<<(std::ostream& os, const EmulatedME0Segment& seg) {
  os << "EmulatedME0Segment: local pos = " << seg.localPosition() << 
    " posErr = (" << sqrt(seg.localPositionError().xx())<<","<<sqrt(seg.localPositionError().yy())<<
    "0,)\n"<<
    "            dir = " << seg.localDirection() <<
    " dirErr = (" << sqrt(seg.localDirectionError().xx())<<","<<sqrt(seg.localDirectionError().yy())<<
    "0,)\n";
  return os;  
}
