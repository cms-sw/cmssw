#include "DataFormats/ParticleFlowReco/interface/PFTrack.h"
#include "Math/GenVector/PositionVector3D.h" 
#include "DataFormats/Math/interface/Point3D.h" 
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;
using namespace std;

const unsigned PFTrack::nMaxTrackingLayers_ = 17;

PFTrack::PFTrack() :
  charge_(0.),
  indexInnermost_(0),
  indexOutermost_(0),
  color_(1) {

  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.reserve(PFTrajectoryPoint::NLayers + nMaxTrackingLayers_);
}


PFTrack::PFTrack(double charge) : 
  charge_(charge), 
  indexInnermost_(0),
  indexOutermost_(0),
  color_(1) {

  // prepare vector of trajectory points for propagated positions
  trajectoryPoints_.reserve(PFTrajectoryPoint::NLayers + nMaxTrackingLayers_);
}
  

PFTrack::PFTrack(const PFTrack& other) :
  charge_(other.charge_), 
  trajectoryPoints_(other.trajectoryPoints_),
  indexInnermost_(other.indexInnermost_),
  indexOutermost_(other.indexOutermost_),
  color_(other.color_)
{}


void PFTrack::addPoint(const PFTrajectoryPoint& trajPt) {
  
  //   cout<<"adding "<<trajPt<<endl;

  if (trajPt.isTrackerLayer()) {
    if (!indexOutermost_) { // first time a measurement is added
      if (trajectoryPoints_.size() < PFTrajectoryPoint::BeamPipeOrEndVertex + 1) {
        PFTrajectoryPoint dummyPt;
        for (unsigned iPt = trajectoryPoints_.size(); iPt < PFTrajectoryPoint::BeamPipeOrEndVertex + 1; iPt++)
          trajectoryPoints_.push_back(dummyPt);
      } else if (trajectoryPoints_.size() > PFTrajectoryPoint::BeamPipeOrEndVertex + 1) {
        // throw an exception here
        //      edm::LogError("PFTrack")<<"trajectoryPoints_.size() is too large = " 
        //                              <<trajectoryPoints_.size()<<"\n";
      }
      indexOutermost_ = indexInnermost_ = PFTrajectoryPoint::BeamPipeOrEndVertex + 1;
    } else 
      indexOutermost_++;
  }
  // Use push_back instead of insert in order to gain time
  trajectoryPoints_.push_back(trajPt);

  //   cout<<"adding point "<<*this<<endl;
}


void PFTrack::calculatePositionREP() {
  
  //for(unsigned i=0; i<trajectoryPoints_.size(); i++) {
  //  trajectoryPoints_[i].calculatePositionREP();
  //}
}

 
const reco::PFTrajectoryPoint& PFTrack::extrapolatedPoint(unsigned layerid) const {
  
  if( layerid >= reco::PFTrajectoryPoint::NLayers ||
      nTrajectoryMeasurements() + layerid >= trajectoryPoints_.size() ) {

    // cout<<(*this)<<endl;
    // cout<<"lid "<<layerid<<" "<<nTrajectoryMeasurements()<<" "<<trajectoryPoints_.size()<<endl;
    
    throw cms::Exception("SizeError")<<"PFRecTrack::extrapolatedPoint: cannot access "
				     <<layerid
				     <<" #traj meas = "<<nTrajectoryMeasurements()
				     <<" #traj points = "<<trajectoryPoints_.size()
				     <<endl
				     <<(*this);
    // assert(0);
  }
  if (layerid < indexInnermost_)
    return trajectoryPoints_[ layerid ];
  else
    return trajectoryPoints_[ nTrajectoryMeasurements() + layerid ];  
}


ostream& reco::operator<<(ostream& out, 
                          const PFTrack& track) {  
  if (!out) return out;  

  const reco::PFTrajectoryPoint& closestApproach = 
    track.trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach);

  out<<"Track charge = "<<track.charge() 
     <<", Pt = "<<closestApproach.momentum().Pt() 
     <<", P = "<<closestApproach.momentum().P()<<endl
     <<"\tR0 = "<<closestApproach.position().Rho()
     <<" Z0 = "<<closestApproach.position().Z()<<endl
     <<"\tnumber of tracker measurements = " 
     <<track.nTrajectoryMeasurements()<<endl;
  for(unsigned i=0; i<track.trajectoryPoints_.size(); i++) 
    out<<track.trajectoryPoints_[i]<<endl;
  

  return out;
}
