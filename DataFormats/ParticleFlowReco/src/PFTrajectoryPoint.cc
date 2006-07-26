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
  detId_(detId),
  layer_(layer),
  posxyz_(posxyz),
  momentum_(momentum) 
{ 
  posrep_.SetCoordinates(posxyz_.Rho(), posxyz_.Eta(), posxyz_.Phi()); 
}


PFTrajectoryPoint::PFTrajectoryPoint(const PFTrajectoryPoint& other) :
  detId_(other.detId_), 
  layer_(other.layer_), 
  posxyz_(other.posxyz_), 
  posrep_(other.posrep_),
  momentum_(other.momentum_) { }


PFTrajectoryPoint::~PFTrajectoryPoint() 
{}


const PFTrajectoryPoint::REPPoint& PFTrajectoryPoint::getPositionREP() {
  if(posrep_ == REPPoint())
    posrep_.SetCoordinates(posxyz_.Rho(), posxyz_.Eta(), posxyz_.Phi());
  return posrep_;
}


std::ostream& reco::operator<<(std::ostream& out, 
			       const reco::PFTrajectoryPoint& trajPoint) {
  if(!out) return out;

  const math::XYZPoint& posxyz = trajPoint.getPositionXYZ();

  out << "Trajectory point id = " << trajPoint.getDetId()
      << ", layer = " << trajPoint.getLayer()
      << ", Pt = " << trajPoint.getMomentum().Pt() 
      << ", Eta = " << posxyz.Eta()
      << ", R = " << posxyz.Rho() << ", Z = " << posxyz.Z() << std::endl;
  
  return out;
}
