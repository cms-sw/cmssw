#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;


PFRecTrack::PFRecTrack() :
  PFTrack(),
  algoType_(PFRecTrack::Unknown) {}


PFRecTrack::PFRecTrack(double charge, AlgoType_t algoType) : 
  PFTrack(charge), 
  algoType_(algoType) {}
  

PFRecTrack::PFRecTrack(const PFRecTrack& other) :
  PFTrack(other), 
  algoType_(other.algoType_) {}

std::ostream& reco::operator<<(std::ostream& out, 
			       const PFRecTrack& track) {  
  if (!out) return out;  

  const reco::PFTrajectoryPoint& closestApproach = 
    track.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out << "Track charge = " << track.charge() 
      << ", type = " << track.algoType()
      << ", Pt = " << closestApproach.momentum().Pt() 
      << ", P = " << closestApproach.momentum().P() << std::endl
      << "\tR0 = " << closestApproach.positionXYZ().Rho()
      <<" Z0 = " << closestApproach.positionXYZ().Z() << std::endl
      << "\tnumber of tracker measurements = " 
      << track.nTrajectoryMeasurements() << std::endl;

  return out;
}
