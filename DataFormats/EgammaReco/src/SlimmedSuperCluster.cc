#include "DataFormats/EgammaReco/interface/SlimmedSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Math/GenVector/PositionVector3D.h"

reco::SlimmedSuperCluster::SlimmedSuperCluster(const reco::SuperCluster& sc, float trkIso)
    : correctedEnergy_(sc.correctedEnergy()),
      rawEnergy_(sc.rawEnergy()),
      preshowerEnergy_(sc.preshowerEnergy()),
      rho_(sc.position().rho()),
      eta_(sc.eta()),
      phi_(sc.phi()),
      trkIso_(trkIso) {
  clusterSeedIds_.push_back(sc.seed()->seed());
  for (const auto& clus : sc.clusters()) {
    if (clus != sc.seed()) {
      clusterSeedIds_.push_back(clus->seed());
    }
  }
}

math::XYZPoint reco::SlimmedSuperCluster::position() const {
  ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<float>> pos;
  pos.SetRho(rho_);
  pos.SetEta(eta_);
  pos.SetPhi(phi_);
  return math::XYZPoint(pos.x(), pos.y(), pos.z());
}
