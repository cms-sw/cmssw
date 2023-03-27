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
                const HgcalClusterFilterMask& availableLayerClusters,
                std::vector<float>& layerClustersMask,
                hgcal::RecHitTools& rhtools) const override {
      auto filteredLayerClusters = std::make_unique<HgcalClusterFilterMask>();
      for (auto const& cl : availableLayerClusters) {
        auto const& layerCluster = layerClusters[cl.first];
        auto const& haf = layerCluster.hitsAndFractions();
        auto layerId = rhtools.getLayerWithOffset(haf[0].first);
        if (find(algo_number_.begin(), algo_number_.end(), layerCluster.algo()) != algo_number_.end() 
         and layerId <= max_layerId_ and layerId >= min_layerId_ and
            haf.size() <= max_cluster_size_ and
            (haf.size() >= min_cluster_size_ or !(rhtools.isSilicon(haf[0].first)))) {
          filteredLayerClusters->emplace_back(cl);
        } else {
          layerClustersMask[cl.first] = 0.;
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
