// Authors: Marco Rovere - marco.rovere@cern.ch, Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2020

#ifndef RecoHGCal_TICL_ClusterFilterByAlgoAndSizeAndLayerRange_H__
#define RecoHGCal_TICL_ClusterFilterByAlgoAndSizeAndLayerRange_H__

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "ClusterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl {
  class ClusterFilterByAlgoAndSizeAndLayerRange final : public ClusterFilterBase {
  public:
    ClusterFilterByAlgoAndSizeAndLayerRange(const edm::ParameterSet& ps)
        : ClusterFilterBase(ps),
          algo_number_(ps.getParameter<std::vector<int>>("algo_number")),
          min_cluster_size_(ps.getParameter<int>("min_cluster_size")),
          max_cluster_size_(ps.getParameter<int>("max_cluster_size")),
          min_layerId_(ps.getParameter<int>("min_layerId")),
          max_layerId_(ps.getParameter<int>("max_layerId")) {}
    ~ClusterFilterByAlgoAndSizeAndLayerRange() override{};

    void filter(const std::vector<reco::CaloCluster>& layerClusters,
                std::vector<float>& layerClustersMask,
                hgcal::RecHitTools& rhtools) const override {
      for (size_t i = 0; i < layerClusters.size(); i++) {
        auto const& layerCluster = layerClusters[i];
        auto const& haf = layerCluster.hitsAndFractions();
        auto layerId = rhtools.getLayerWithOffset(haf[0].first);
        if (find(algo_number_.begin(), algo_number_.end(), layerCluster.algo()) == algo_number_.end() or
            layerId > max_layerId_ or layerId < min_layerId_ or haf.size() > max_cluster_size_ or
            (haf.size() < min_cluster_size_ and rhtools.isSilicon(haf[0].first))) {
          layerClustersMask[i] = 0.;
        }
      }
    }

  private:
    std::vector<int> algo_number_;
    unsigned int min_cluster_size_;
    unsigned int max_cluster_size_;
    unsigned int min_layerId_;
    unsigned int max_layerId_;
  };
}  // namespace ticl

#endif
