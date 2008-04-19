#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "Math/GenVector/PositionVector3D.h" 
#include "DataFormats/Math/interface/Point3D.h" 
// #include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;


PFRecTrack::PFRecTrack() :
  PFTrack(),
  algoType_(PFRecTrack::Unknown),
  pfBremVec_(0)
{}



PFRecTrack::PFRecTrack(double charge, 
                       AlgoType_t algoType, 
                       int trackId, 
                       const reco::TrackRef& trackRef ) : 
  PFTrack(charge), 
  algoType_(algoType),
  trackId_(trackId), 
  trackRef_(trackRef),
  pfBremVec_(0)
{}



PFRecTrack::PFRecTrack(double charge, AlgoType_t algoType) : 
  PFTrack(charge), 
  algoType_(algoType),
  trackId_(-1),
  pfBremVec_(0)
{}
  
void PFRecTrack::addBrem(const reco::PFBrem& brem){
  pfBremVec_.push_back(brem);
}

std::ostream& reco::operator<<(std::ostream& out, 
                               const PFRecTrack& track) {  
  if (!out) return out;  
  
  const reco::PFTrajectoryPoint& closestApproach = 
    track.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);
  
  out << "Reco track charge = " << track.charge() 
      << ", type = " << track.algoType()
      << ", Pt = " << closestApproach.momentum().Pt() 
      << ", P = " << closestApproach.momentum().P() << std::endl
      << "\tR0 = " << closestApproach.positionXYZ().Rho()
      <<" Z0 = " << closestApproach.positionXYZ().Z() << std::endl
      << "\tnumber of tracker measurements = " 
      << track.nTrajectoryMeasurements() << std::endl
      <<"\tnumber of points total = "
      << track.trajectoryPoints_.size()<< std::endl;

  for(unsigned i=0; i<track.trajectoryPoints_.size(); i++) 
    out<<track.trajectoryPoints_[i]<<std::endl;


  return out;
}
