#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

using namespace reco;

PFRecTrack::PFRecTrack() :
  charge_(0.),
  algoType_(PFRecTrack::Unknown),
  indexInnermost_(0),
  indexOutermost_(0),
  doPropagation_(true),
  color_(1) {

  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.resize(PFTrajectoryPoint::NLayers);
}


PFRecTrack::PFRecTrack(double charge, AlgoType_t algoType) : 
  charge_(charge), 
  algoType_(algoType), 
  indexInnermost_(0),
  indexOutermost_(0),
  doPropagation_(true),
  color_(1) {

  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.resize(PFTrajectoryPoint::NLayers);
}
  

PFRecTrack::PFRecTrack(const PFRecTrack& other) :
  charge_(other.charge_), 
  algoType_(other.algoType_),
  trajectoryPoints_(other.trajectoryPoints_),
  indexInnermost_(other.indexInnermost_),
  indexOutermost_(other.indexOutermost_),
  doPropagation_(other.doPropagation_),
  color_(other.color_)
{}


void PFRecTrack::addMeasurement(const PFTrajectoryPoint& measurement) {
  
  if (!indexOutermost_) // first time a measurement is added
    indexOutermost_ = indexInnermost_ = PFTrajectoryPoint::BeamPipe + 1;
  std::vector< PFTrajectoryPoint >::iterator it = 
    trajectoryPoints_.begin() + indexOutermost_;
  trajectoryPoints_.insert(it, PFTrajectoryPoint(measurement));
  indexOutermost_++;
  
  // COLIN: vector::insert is a very time consuming operation (linear)
}


void PFRecTrack::CalculatePositionREP() {
  
  for(unsigned i=0; i<trajectoryPoints_.size(); i++) {
    trajectoryPoints_[i].CalculatePositionREP();
  }
}

 
const reco::PFTrajectoryPoint& PFRecTrack::getExtrapolatedPoint(unsigned layerid) const {
  
  if( layerid >= reco::PFTrajectoryPoint::NLayers ) {
    assert(0);
  }

  return trajectoryPoints_[ getNTrajectoryMeasurements() + layerid ];  
}



std::ostream& reco::operator<<(std::ostream& out, 
			       const PFRecTrack& track) {  
  if (!out) return out;  
//   if (!track.IsPropagated()) track.Propagate();

  const reco::PFTrajectoryPoint& closestApproach = 
    track.getTrajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out << "Track charge = " << track.getCharge() 
      << ", type = " << track.getAlgoType()
      << ", Pt = " << closestApproach.getMomentum().Pt() 
      << ", P = " << closestApproach.getMomentum().P() << std::endl
      << "\tR0 = " << closestApproach.getPositionXYZ().Rho()
      <<" Z0 = " << closestApproach.getPositionXYZ().Z() << std::endl
      << "\tnumber of tracker measurements = " 
      << track.getNTrajectoryMeasurements() << std::endl;

  return out;
}
