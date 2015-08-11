/** \file ME0egment.cc
 *
 *  $Date: 2014/02/04 12:41:33 $
 *  \author Marcello Maggi
 */

#include <DataFormats/GEMRecHit/interface/ME0Segment.h>
#include <iostream>

namespace {
  // define a Super Layer Id from the first layer of the first rechits, and put to first layer
  inline
  DetId buildDetId(ME0DetId id) {
    return ME0DetId(id.region(),1,id.chamber(),id.roll());
  }

}

ME0Segment::ME0Segment(const std::vector<const ME0RecHit*>& proto_segment, LocalPoint origin, 
	LocalVector direction, AlgebraicSymMatrix errors, double chi2) : 
  RecSegment(buildDetId(proto_segment.front()->me0Id())),
  theOrigin(origin), 
  theLocalDirection(direction), theCovMatrix(errors), theChi2(chi2){

  for(unsigned int i=0; i<proto_segment.size(); ++i)
    theME0RecHits.push_back(*proto_segment[i]);
}

ME0Segment::~ME0Segment() {}

std::vector<const TrackingRecHit*> ME0Segment::recHits() const{
  std::vector<const TrackingRecHit*> pointersOfRecHits;
  for (std::vector<ME0RecHit>::const_iterator irh = theME0RecHits.begin(); irh!=theME0RecHits.end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  return pointersOfRecHits;
}

std::vector<TrackingRecHit*> ME0Segment::recHits() {
  
  std::vector<TrackingRecHit*> pointersOfRecHits;
  for (std::vector<ME0RecHit>::iterator irh = theME0RecHits.begin(); irh!=theME0RecHits.end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  return pointersOfRecHits;
}

LocalError ME0Segment::localPositionError() const {
  return LocalError(theCovMatrix[2][2], theCovMatrix[2][3], theCovMatrix[3][3]);
}

LocalError ME0Segment::localDirectionError() const {
  return LocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]); 
}


AlgebraicVector ME0Segment::parameters() const {
  // For consistency with DT and CSC  and what we require for the TrackingRecHit interface,
  // the order of the parameters in the returned vector should be (dx/dz, dy/dz, x, z)
  
  AlgebraicVector result(4);

  result[0] = theLocalDirection.x()/theLocalDirection.z();
  result[1] = theLocalDirection.y()/theLocalDirection.z();    
  result[2] = theOrigin.x();
  result[3] = theOrigin.y();

  return result;
}


AlgebraicMatrix ME0Segment::projectionMatrix() const {
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

float ME0Segment::time() const {
  float averageTime=0;
  for (std::vector<ME0RecHit>::const_iterator itRH = theME0RecHits.begin();
       itRH != theME0RecHits.end(); ++itRH) {
    const  ME0RecHit *recHit = &(*itRH);
    averageTime+=recHit->tof();
  }
  averageTime=averageTime/(theME0RecHits.size());
  return averageTime;
}

//
void ME0Segment::print() const {
  std::cout << *this << std::endl;
}

std::ostream& operator<<(std::ostream& os, const ME0Segment& seg) {
  os << "ME0Segment: local pos = " << seg.localPosition() << 
    " posErr = (" << sqrt(seg.localPositionError().xx())<<","<<sqrt(seg.localPositionError().yy())<<
    "0,)\n"<<
    "            dir = " << seg.localDirection() <<
    " dirErr = (" << sqrt(seg.localDirectionError().xx())<<","<<sqrt(seg.localDirectionError().yy())<<
    "0,)\n"<<
    "            chi2/ndf = " << seg.chi2()/double(seg.degreesOfFreedom()) << 
    " #rechits = " << seg.specificRecHits().size();
  return os;  
}

