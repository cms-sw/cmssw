#ifndef RECOHGCAL_TICL_TRACKSTERSPCA_H
#define RECOHGCAL_TICL_TRACKSTERSPCA_H

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include <vector>
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {
  /**
   * Computes the trackster raw energy, barycenter, timing, PCA.
   * PCA is computed using layer cluster positions (and energies if energyWeight=True)
   * In case clean=True: Only the most energetic layer cluster on each layer is considered,
   *   and only layers in range around the most energetic layer cluster
   * \param z_limit_em Limit between EM and HAD section (in absolute value)
   * \param energyWeight Compute energy-weighted barycenter and PCA
   * \param clean Use the PCA cleaning algorithm. 
   * \param minLayer Number of layers to consider for cleaned PCA behind the most energetic LC
   * \param maxLayer Number of layers to consider for cleaned PCA after the most energetic LC
   */
  void assignPCAtoTracksters(std::vector<Trackster> &tracksters,
                             const std::vector<reco::CaloCluster> &layerClusters,
                             const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                             double z_limit_em,
                             hgcal::RecHitTools const &rhTools,
                             bool computeLocalTime = false,
                             bool energyWeight = true,
                             bool clean = false,
                             int minLayer = 10,
                             int maxLayer = 10);
  std::pair<float, float> computeLocalTracksterTime(const Trackster &trackster,
                                                    const std::vector<reco::CaloCluster> &layerClusters,
                                                    const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                                    const Eigen::Vector3f &barycenter,
                                                    size_t N);
  std::pair<float, float> computeTracksterTime(const Trackster &trackster,
                                               const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                               size_t N);

  inline unsigned getLayerFromLC(const reco::CaloCluster &LC, const hgcal::RecHitTools &rhtools) {
    std::vector<std::pair<DetId, float>> thisclusterHits = LC.hitsAndFractions();
    auto layer = rhtools.getLayerWithOffset(thisclusterHits[0].first);
    return layer;
  }

  // Sort the layer clusters in the given trackster in bins of layer. Returns : vector[index=layer, value=vector[LC index]]]
  inline std::vector<std::vector<unsigned>> sortByLayer(const Trackster &ts,
                                                        const std::vector<reco::CaloCluster> &layerClusters,
                                                        const hgcal::RecHitTools &rhtools) {
    size_t N = ts.vertices().size();

    std::vector<std::vector<unsigned>> result;
    result.resize(rhtools.lastLayer() + 1);

    for (unsigned i = 0; i < N; ++i) {
      const auto &thisLC = layerClusters[ts.vertices(i)];
      auto layer = getLayerFromLC(thisLC, rhtools);
      result[layer].push_back(i);
    }
    return result;
  }
}  // namespace ticl
#endif
