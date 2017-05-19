/** \file GEMegment.cc
 *
 *  \author Piet Verwilligen
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"
#include <iostream>

namespace {
  // create reference GEM Chamber ID for segment
  inline
  DetId buildDetId(GEMDetId id) {
    return GEMDetId(id.superChamberId());
  }
}

class ProjectionMatrixDiag {
  // Aider class to make the return of the projection Matrix thread-safe
protected:
  AlgebraicMatrix theProjectionMatrix;
public:
  ProjectionMatrixDiag() : theProjectionMatrix(4,5,0) {
    theProjectionMatrix[0][1] = 1;
    theProjectionMatrix[1][2] = 1;
    theProjectionMatrix[2][3] = 1;
    theProjectionMatrix[3][4] = 1;
  }
  const AlgebraicMatrix &getMatrix() const {
    return (theProjectionMatrix);
  }
};


GEMSegment::GEMSegment(const std::vector<const GEMRecHit*>& proto_segment, const LocalPoint& origin, 
	const LocalVector& direction, const AlgebraicSymMatrix& errors, double chi2) :
  RecSegment(buildDetId(proto_segment.front()->gemId())), 
  theOrigin(origin), 
  theLocalDirection(direction), theCovMatrix(errors), theChi2(chi2){
  theTimeValue = 0.0;
  theTimeUncrt = 0.0;
  theBX = -10.0;
  for(unsigned int i=0; i<proto_segment.size(); ++i)
    theGEMRecHits.push_back(*proto_segment[i]);
}

GEMSegment::GEMSegment(const std::vector<const GEMRecHit*>& proto_segment, const LocalPoint& origin, 
		       const LocalVector& direction, const AlgebraicSymMatrix& errors, double chi2, float bx) : 
  RecSegment(buildDetId(proto_segment.front()->gemId())),
  theOrigin(origin), 
  theLocalDirection(direction), theCovMatrix(errors), theChi2(chi2){
  theTimeValue = 0.0;
  theTimeUncrt = 0.0;
  theBX = bx;
  for(unsigned int i=0; i<proto_segment.size(); ++i)
    theGEMRecHits.push_back(*proto_segment[i]);
}

GEMSegment::GEMSegment(const std::vector<const GEMRecHit*>& proto_segment, const LocalPoint& origin, 
		       const LocalVector& direction, const AlgebraicSymMatrix& errors, double chi2, double time, double timeErr) : 
  RecSegment(buildDetId(proto_segment.front()->gemId())),
  theOrigin(origin), 
  theLocalDirection(direction), theCovMatrix(errors), theChi2(chi2){
  theTimeValue = time;
  theTimeUncrt = timeErr;
  theBX = -10.0;
  for(unsigned int i=0; i<proto_segment.size(); ++i)
    theGEMRecHits.push_back(*proto_segment[i]);
}

GEMSegment::~GEMSegment() {}

std::vector<const TrackingRecHit*> GEMSegment::recHits() const{
  std::vector<const TrackingRecHit*> pointersOfRecHits;
  for (std::vector<GEMRecHit>::const_iterator irh = theGEMRecHits.begin(); irh!=theGEMRecHits.end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  return pointersOfRecHits;
}

std::vector<TrackingRecHit*> GEMSegment::recHits() {
  
  std::vector<TrackingRecHit*> pointersOfRecHits;
  for (std::vector<GEMRecHit>::iterator irh = theGEMRecHits.begin(); irh!=theGEMRecHits.end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  return pointersOfRecHits;
}

LocalError GEMSegment::localPositionError() const {
  return LocalError(theCovMatrix[2][2], theCovMatrix[2][3], theCovMatrix[3][3]);
}

LocalError GEMSegment::localDirectionError() const {
  return LocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]); 
}


AlgebraicVector GEMSegment::parameters() const {
  // For consistency with DT and CSC  and what we require for the TrackingRecHit interface,
  // the order of the parameters in the returned vector should be (dx/dz, dy/dz, x, z)
  
  AlgebraicVector result(4);

  if(theLocalDirection.z() != 0)
  {
  result[0] = theLocalDirection.x()/theLocalDirection.z();
  result[1] = theLocalDirection.y()/theLocalDirection.z();    
  }
  result[2] = theOrigin.x();
  result[3] = theOrigin.y();

  return result;
}

AlgebraicMatrix GEMSegment::projectionMatrix() const {
  static const ProjectionMatrixDiag theProjectionMatrix;
  return (theProjectionMatrix.getMatrix());
}

//
void GEMSegment::print() const {
  LogDebug("GEMSegment") << *this;

}

std::ostream& operator<<(std::ostream& os, const GEMSegment& seg) {
  os << "GEMSegment: local pos = " << seg.localPosition() << 
    " posErr = (" << sqrt(seg.localPositionError().xx())<<","<<sqrt(seg.localPositionError().yy())<<
    "0,)\n"<<
    "            dir = " << seg.localDirection() <<
    " dirErr = (" << sqrt(seg.localDirectionError().xx())<<","<<sqrt(seg.localDirectionError().yy())<<
    "0,)\n"<<
    "            chi2/ndf = " << ((seg.degreesOfFreedom() != 0.) ? seg.chi2()/double(seg.degreesOfFreedom()) :0 ) << 
    " #rechits = " << seg.specificRecHits().size()<<
    " bx = "<< seg.bunchX() <<
    " time = "<< seg.time() << " +/- " << seg.timeErr() << " ns";

  return os;  
}

