/** \file GEMCSCSegment.cc
 *
 *  Based on CSCSegment class
 *  \author Raffaella Radogna
 */

#include <DataFormats/GEMRecHit/interface/GEMCSCSegment.h>
#include <iostream>


namespace {
  // Get CSCDetId from one of the rechits, but then remove the layer part so it's a _chamber_ id
  inline
  DetId buildDetId(CSCDetId id) {
    return CSCDetId (id.endcap(),id.station(),id.ring(),id.chamber(),0);
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




GEMCSCSegment::GEMCSCSegment(const CSCSegment* csc_segment, const std::vector<const GEMRecHit*> gem_rhs, LocalPoint origin, LocalVector direction, AlgebraicSymMatrix errors, double chi2) : 

  RecSegment(buildDetId(csc_segment->cscDetId())),
  theOrigin(origin), 
  theLocalDirection(direction), theCovMatrix(errors), theChi2(chi2) {

  for(unsigned int i=0; i<gem_rhs.size(); ++i) {
    theGEMRecHits.push_back((*gem_rhs[i]));
  }
  theCSCSegment = *csc_segment;
  // LogDebug
  edm::LogVerbatim("GEMCSCSegment")<< "[GEMCSCSegment :: ctor] CSCDetId: " << csc_segment->cscDetId() << " CSC RecHits: " <<csc_segment->specificRecHits().size()
				   << " GEM RecHits: " << gem_rhs.size()<<"\n" //  << " Fit chi2: "<<chi2<<" Position: "<<origin<<" Direction: "<<direction
				   << "    CSC Segment Details: \n"<<*csc_segment<<"\n"
				   << " GEMCSC Segment Details: \n"<<*this<<"\n"
				   << "[GEMCSCSegment :: ctor] ------------------------------------------------------------";
}


GEMCSCSegment::~GEMCSCSegment() {}


std::vector<const TrackingRecHit*> GEMCSCSegment::recHits() const{

  std::vector<const TrackingRecHit*> pointersOfRecHits;
  for (std::vector<GEMRecHit>::const_iterator irh = theGEMRecHits.begin(); irh!=theGEMRecHits.end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  for (std::vector<CSCRecHit2D>::const_iterator irh = theCSCSegment.specificRecHits().begin(); irh!=theCSCSegment.specificRecHits().end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  return pointersOfRecHits;
}

std::vector<TrackingRecHit*> GEMCSCSegment::recHits() {

  std::vector<TrackingRecHit*> pointersOfRecHits;
  for (std::vector<GEMRecHit>::iterator irh = theGEMRecHits.begin(); irh!=theGEMRecHits.end(); ++irh) {
    pointersOfRecHits.push_back(&(*irh));
  }
  return pointersOfRecHits;
}


LocalError GEMCSCSegment::localPositionError() const {
  return LocalError(theCovMatrix[2][2], theCovMatrix[2][3], theCovMatrix[3][3]);
}


LocalError GEMCSCSegment::localDirectionError() const {
  return LocalError(theCovMatrix[0][0], theCovMatrix[0][1], theCovMatrix[1][1]); 
}


AlgebraicVector GEMCSCSegment::parameters() const {
  // For consistency with DT, CSC and what we require for the TrackingRecHit interface,
  // the order of the parameters in the returned vector should be (dx/dz, dy/dz, x, z)
  
  AlgebraicVector result(4);
  if(theLocalDirection.z()!=0) {
    result[0] = theLocalDirection.x()/theLocalDirection.z();
    result[1] = theLocalDirection.y()/theLocalDirection.z();    
  }
  result[2] = theOrigin.x();
  result[3] = theOrigin.y();
  return result;
}

AlgebraicMatrix GEMCSCSegment::projectionMatrix() const {
  static const ProjectionMatrixDiag theProjectionMatrix;
  return (theProjectionMatrix.getMatrix());
}




std::ostream& operator<<(std::ostream& os, const GEMCSCSegment& seg) {
  os << "GEMCSCSegment: local pos = " << seg.localPosition() << 
    " posErr = (" << sqrt(seg.localPositionError().xx())<<","<<sqrt(seg.localPositionError().yy())<<
    "0,)\n"<<
    "            dir = " << seg.localDirection() <<
    " dirErr = (" << sqrt(seg.localDirectionError().xx())<<","<<sqrt(seg.localDirectionError().yy())<<
    "0,)\n"<<
    "            chi2/ndf = " << ((seg.degreesOfFreedom()!=0)?(seg.chi2()/double(seg.degreesOfFreedom())):0.0) << 
    " #rechits = " << seg.nRecHits();
  return os;  
}


