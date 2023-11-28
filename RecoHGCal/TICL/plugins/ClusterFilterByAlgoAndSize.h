// Authors: Marco Rovere - marco.rovere@cern.ch, Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018

#ifndef RecoHGCal_TICL_ClusterFilterByAlgoAndSize_H__
#define RecoHGCal_TICL_ClusterFilterByAlgoAndSize_H__

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "ClusterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl {
  class ClusterFilterByAlgoAndSize final : public ClusterFilterBase {
  public:
    ClusterFilterByAlgoAndSize(const edm::ParameterSet& ps)
        : ClusterFilterBase(ps),
          algo_number_(ps.getParameter<std::vector<int>>("algo_number")),
          min_cluster_size_(ps.getParameter<int>("min_cluster_size")),
          max_cluster_size_(ps.getParameter<int>("max_cluster_size")) {}
    ~ClusterFilterByAlgoAndSize() override{};

    void filter(const std::vector<reco::CaloCluster>& layerClusters,
                const TICLClusterFilterMask& availableLayerClusters,
                std::vector<float>& layerClustersMask,
                hgcal::RecHitTools& rhtools) const override {
      auto filteredLayerClusters = std::make_unique<TICLClusterFilterMask>();
      for (auto const& cl : availableLayerClusters) {
        auto const& layerCluster = layerClusters[cl.first];
        if (find(algo_number_.begin(), algo_number_.end(), layerCluster.algo()) != algo_number_.end() and
            layerCluster.hitsAndFractions().size() <= max_cluster_size_ and
            (layerCluster.hitsAndFractions().size() >= min_cluster_size_ or
             (!(rhtools.isSilicon(layerCluster.hitsAndFractions()[0].first))))) {
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
  };
}  // namespace ticl

#endif
