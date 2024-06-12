#ifndef RECOHGCAL_TICL_TRACKSTERSPCA_H
#define RECOHGCAL_TICL_TRACKSTERSPCA_H

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include <vector>

namespace ticl {
  void assignPCAtoTracksters(std::vector<Trackster> &,
                             const std::vector<reco::CaloCluster> &,
                             const edm::ValueMap<std::pair<float, float>> &,
                             double,
                             bool computeLocalTime = false,
                             bool energyWeight = true);
  std::pair<float, float> computeLocalTracksterTime(const Trackster &trackster,
                                                    const std::vector<reco::CaloCluster> &layerClusters,
                                                    const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                                    const Eigen::Vector3d &barycenter,
                                                    size_t N);
  std::pair<float, float> computeTracksterTime(const Trackster &trackster,
                                               const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                               size_t N);
}  // namespace ticl
#endif
