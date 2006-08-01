#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;

const unsigned PFRecTrack::nMaxTrackingLayers_ = 17;

PFRecTrack::PFRecTrack() :
  charge_(0.),
  algoType_(PFRecTrack::Unknown),
  indexInnermost_(0),
  indexOutermost_(0),
  color_(1) {

  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.reserve(PFTrajectoryPoint::NLayers + nMaxTrackingLayers_);
}


PFRecTrack::PFRecTrack(double charge, AlgoType_t algoType) : 
  charge_(charge), 
  algoType_(algoType), 
  indexInnermost_(0),
  indexOutermost_(0),
  color_(1) {

  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.reserve(PFTrajectoryPoint::NLayers + nMaxTrackingLayers_);
}
  

PFRecTrack::PFRecTrack(const PFRecTrack& other) :
  charge_(other.charge_), 
  algoType_(other.algoType_),
  trajectoryPoints_(other.trajectoryPoints_),
  indexInnermost_(other.indexInnermost_),
  indexOutermost_(other.indexOutermost_),
  color_(other.color_)
{}


void PFRecTrack::addPoint(const PFTrajectoryPoint& trajPt) {
  
  if (trajPt.isTrackerLayer() && !indexOutermost_) { // first time a measurement is added
    if (trajectoryPoints_.size() < PFTrajectoryPoint::BeamPipe + 1) {
      PFTrajectoryPoint dummyPt;
      for (unsigned iPt = trajectoryPoints_.size(); iPt < PFTrajectoryPoint::BeamPipe + 1; iPt++)
	trajectoryPoints_.push_back(dummyPt);
    } else if (trajectoryPoints_.size() > PFTrajectoryPoint::BeamPipe + 1) {
      edm::LogError("PFRecTrack") << "trajectoryPoints_.size() is too large = " 
				  << trajectoryPoints_.size() << "\n";
    }
    indexOutermost_ = indexInnermost_ = PFTrajectoryPoint::BeamPipe + 1;
  }
  // Use push_back instead of insert in order to gain time
//   std::vector< PFTrajectoryPoint >::iterator it = 
//     trajectoryPoints_.begin() + indexOutermost_;
//   trajectoryPoints_.insert(it, PFTrajectoryPoint(measurement));
  trajectoryPoints_.push_back(trajPt);
  indexOutermost_++;
}


void PFRecTrack::calculatePositionREP() {
  
  for(unsigned i=0; i<trajectoryPoints_.size(); i++) {
    trajectoryPoints_[i].calculatePositionREP();
  }
}

 
const reco::PFTrajectoryPoint& PFRecTrack::extrapolatedPoint(unsigned layerid) const {
  
  if( layerid >= reco::PFTrajectoryPoint::NLayers ) {
    assert(0);
  }
  if (layerid < indexInnermost_)
    return trajectoryPoints_[ layerid ];
  else
    return trajectoryPoints_[ nTrajectoryMeasurements() + layerid ];  
}


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
