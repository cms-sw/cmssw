#ifndef RecoEgamma_EgammaTools_HGCalClusterTools_h
#define RecoEgamma_EgammaTools_HGCalClusterTools_h

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include <vector>

class HGCalClusterTools {
public:
  enum class EType { ET, ENERGY };

  static float energyInCone(const float eta,
                            const float phi,
                            const std::vector<reco::CaloCluster>& layerClusters,
                            const float minDR,
                            const float maxDR,
                            const float minEt,
                            const float minEnergy,
                            const std::vector<DetId::Detector>& subDets,
                            const HGCalClusterTools::EType& eType = EType::ENERGY);

  static float hadEnergyInCone(const float eta,
                               const float phi,
                               const std::vector<reco::CaloCluster>& layerClusters,
                               const float minDR,
                               const float maxDR,
                               const float minEt,
                               const float minEnergy,
                               const HGCalClusterTools::EType& eType = EType::ENERGY) {
    return energyInCone(
        eta, phi, layerClusters, minDR, maxDR, minEt, minEnergy, {DetId::HGCalHSi, DetId::HGCalHSc}, eType);
  }
  static float emEnergyInCone(const float eta,
                              const float phi,
                              const std::vector<reco::CaloCluster>& layerClusters,
                              const float minDR,
                              const float maxDR,
                              const float minEt,
                              const float minEnergy,
                              const HGCalClusterTools::EType& eType = EType::ENERGY) {
    return energyInCone(eta, phi, layerClusters, minDR, maxDR, minEt, minEnergy, {DetId::HGCalEE}, eType);
  }
};

#endif
