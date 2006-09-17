#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

PFTrajectoryPoint::PFTrajectoryPoint() : 
  isTrackerLayer_(false),
  detId_(0),
  layer_(0),
  posxyz_(math::XYZPoint(0.,0.,0.)),
  posrep_(REPPoint(0.,0.,0.)),
  momentum_(math::XYZTLorentzVector(0., 0., 0., 0.)) { }


PFTrajectoryPoint::PFTrajectoryPoint(unsigned detId,
				     unsigned layer, 
				     const math::XYZPoint& posxyz, 
				     const math::XYZTLorentzVector& momentum) :
  isTrackerLayer_(false),
  detId_(detId),
  layer_(layer),
  posxyz_(posxyz),
  momentum_(momentum) 
{ 
  if (detId) isTrackerLayer_ = true;
  posrep_.SetCoordinates(posxyz_.Rho(), posxyz_.Eta(), posxyz_.Phi()); 
}


PFTrajectoryPoint::PFTrajectoryPoint(const PFTrajectoryPoint& other) :
  isTrackerLayer_(other.isTrackerLayer_),
  detId_(other.detId_), 
  layer_(other.layer_), 
  posxyz_(other.posxyz_), 
  posrep_(other.posrep_),
  momentum_(other.momentum_) { }


PFTrajectoryPoint::~PFTrajectoryPoint() 
{}


bool   PFTrajectoryPoint::operator==(const reco::PFTrajectoryPoint& other) const {
  if( posxyz_ == other.posxyz_ && 
      momentum_ == other.momentum_ ) return true;
  else return false;
}

std::ostream& reco::operator<<(std::ostream& out, 
			       const reco::PFTrajectoryPoint& trajPoint) {
  if(!out) return out;

  const math::XYZPoint& posxyz = trajPoint.positionXYZ();

  out<<"Traj point id = "<<trajPoint.detId()
     <<", layer = "<<trajPoint.layer()
     <<", Eta,Phi = "<<posxyz.Eta()<<","<<posxyz.Phi()
     <<", X,Y = "<<posxyz.X()<<","<<posxyz.Y()
     <<", R,Z = "<<posxyz.R()<<","<<posxyz.Z();
  
  return out;
}
