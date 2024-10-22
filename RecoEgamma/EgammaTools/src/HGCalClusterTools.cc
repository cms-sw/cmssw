#include "RecoEgamma/EgammaTools/interface/HGCalClusterTools.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

float HGCalClusterTools::energyInCone(const float eta,
                                      const float phi,
                                      const std::vector<reco::CaloCluster>& layerClusters,
                                      const float minDR,
                                      const float maxDR,
                                      const float minEt,
                                      const float minEnergy,
                                      const std::vector<DetId::Detector>& subDets,
                                      const HGCalClusterTools::EType& eType) {
  float hadValue = 0.;

  const float minDR2 = minDR * minDR;
  const float maxDR2 = maxDR * maxDR;

  for (auto& clus : layerClusters) {
    if (clus.energy() < minEnergy) {
      continue;
    }

    if (std::find(subDets.begin(), subDets.end(), clus.seed().det()) == subDets.end()) {
      continue;
    }

    float clusEt = clus.energy() * std::sin(clus.position().theta());
    if (clusEt < minEt) {
      continue;
    }

    //this is a prefilter on the clusters before we calculuate
    //the expensive eta() of the cluster
    float dPhi = reco::deltaPhi(phi, clus.phi());
    if (dPhi > maxDR) {
      continue;
    }

    float dR2 = reco::deltaR2(eta, phi, clus.eta(), clus.phi());
    if (dR2 < minDR2 || dR2 > maxDR2) {
      continue;
    }
    switch (eType) {
      case EType::ET:
        hadValue += clusEt;
        break;
      case EType::ENERGY:
        hadValue += clus.energy();
        break;
    }
  }
  return hadValue;
}
