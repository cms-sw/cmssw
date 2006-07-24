#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

using namespace reco;

PFRecTrack::PFRecTrack() :
  charge_(0.),
  algoType_(PFRecTrack::Unknown),
  indexInnermost_(0),
  indexOutermost_(0),
  doPropagation_(true)
{
  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.resize(PFTrajectoryPoint::NLayers);
}


PFRecTrack::PFRecTrack(double charge, AlgoType_t algoType) : 
  charge_(charge), 
  algoType_(algoType), 
  indexInnermost_(0),
  indexOutermost_(0),
  doPropagation_(true)
{
  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.resize(PFTrajectoryPoint::NLayers);
}
  

PFRecTrack::PFRecTrack(const PFRecTrack& other) :
  charge_(other.charge_), 
  algoType_(other.algoType_),
  trajectoryPoints_(other.trajectoryPoints_),
  indexInnermost_(other.indexInnermost_),
  indexOutermost_(other.indexOutermost_),
  doPropagation_(other.doPropagation_)
{}


void PFRecTrack::addMeasurement(const PFTrajectoryPoint& measurement)
{
  if (!indexOutermost_) // first time a measurement is added
    indexOutermost_ = indexInnermost_ = PFTrajectoryPoint::BeamPipe + 1;
  std::vector< PFTrajectoryPoint >::iterator it = 
    trajectoryPoints_.begin() + indexOutermost_;
  trajectoryPoints_.insert(it, PFTrajectoryPoint(measurement));
  indexOutermost_++;
}


std::ostream& reco::operator<<(std::ostream& out, 
			       const PFRecTrack& track) 
{  
  if (!out) return out;  
//   if (!track.IsPropagated()) track.Propagate();

  const reco::PFTrajectoryPoint& closestApproach = 
    track.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out << "Track charge = " << track.charge() 
      << ", type = " << track.algoType()
      << ", Pt = " << closestApproach.momentum().Pt() 
      << ", P = " << closestApproach.momentum().P() << std::endl
      << "\tEta = " << closestApproach.momentum().Eta()
      << ", Phi = " << closestApproach.momentum().Phi() << std::endl
      << "\tR0 = " << closestApproach.xyzPosition().Rho()
      <<" Z0 = " << closestApproach.xyzPosition().Z() << std::endl
      << "\tnumber of tracker measurements = " 
      << track.nTrajectoryMeasurements() << std::endl;

  return out;
}
