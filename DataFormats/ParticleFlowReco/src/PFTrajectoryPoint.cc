#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include <ostream>
#include <algorithm>

using namespace reco;

// To be kept in synch with the enumerator definitions in PFTrajectoryPoint.h file
// Don't consider "Unknown" and "NLayers"
std::string const PFTrajectoryPoint::layerTypeNames[] = {"ClosestApproach",
                                                         "BeamPipeOrEndVertex",
                                                         "PS1",
                                                         "PS2",
                                                         "ECALEntrance",
                                                         "ECALShowerMax",
                                                         "HCALEntrance",
                                                         "HCALExit",
                                                         "HOLayer",
                                                         "VFcalEntrance"};

PFTrajectoryPoint::PFTrajectoryPoint() : isTrackerLayer_(false), detId_(-1), layer_(-1) {}

PFTrajectoryPoint::PFTrajectoryPoint(int detId,
                                     int layer,
                                     const math::XYZPoint& posxyz,
                                     const math::XYZTLorentzVector& momentum)
    : isTrackerLayer_(false), detId_(detId), layer_(layer), posxyz_(posxyz), momentum_(momentum) {
  if (detId)
    isTrackerLayer_ = true;
  posrep_.SetCoordinates(posxyz_.Rho(), posxyz_.Eta(), posxyz_.Phi());
}

PFTrajectoryPoint::PFTrajectoryPoint(const PFTrajectoryPoint& other)
    : isTrackerLayer_(other.isTrackerLayer_),
      detId_(other.detId_),
      layer_(other.layer_),
      posxyz_(other.posxyz_),
      posrep_(other.posrep_),
      momentum_(other.momentum_) {}

PFTrajectoryPoint::~PFTrajectoryPoint() {}

PFTrajectoryPoint::LayerType PFTrajectoryPoint::layerTypeByName(const std::string& name) {
  LayerType size = NLayers;
  int index = std::find(layerTypeNames, layerTypeNames + size, name) - layerTypeNames;
  if (index == size) {
    return Unknown;  // better this or throw() ?
  }
  return LayerType(index);
}

bool PFTrajectoryPoint::operator==(const reco::PFTrajectoryPoint& other) const {
  if (posxyz_ == other.posxyz_ && momentum_ == other.momentum_)
    return true;
  else
    return false;
}

std::ostream& reco::operator<<(std::ostream& out, const reco::PFTrajectoryPoint& trajPoint) {
  if (!out)
    return out;

  const math::XYZPoint& posxyz = trajPoint.position();

  out << "Traj point id = " << trajPoint.detId() << ", layer = " << trajPoint.layer() << ", Eta,Phi = " << posxyz.Eta()
      << "," << posxyz.Phi() << ", X,Y = " << posxyz.X() << "," << posxyz.Y() << ", R,Z = " << posxyz.Rho() << ","
      << posxyz.Z() << ", E,Pt = " << trajPoint.momentum().E() << "," << trajPoint.momentum().Pt();

  return out;
}
