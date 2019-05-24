// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 11/2018

#ifndef RecoHGCal_TICL_ClusterFilterByAlgoOrSize_H__
#define RecoHGCal_TICL_ClusterFilterByAlgoOrSize_H__

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "ClusterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl{
class ClusterFilterByAlgoOrSize final : public ClusterFilterBase {
 public:
  ClusterFilterByAlgoOrSize(const edm::ParameterSet& ps)
      : ClusterFilterBase(ps),
        algo_number_(ps.getParameter<int>("algo_number")),
        max_cluster_size_(ps.getParameter<int>("max_cluster_size")) {}
  ~ClusterFilterByAlgoOrSize() override {};

  std::unique_ptr<HgcalClusterFilterMask> filter(
      const std::vector<reco::CaloCluster>& layerClusters,
      const HgcalClusterFilterMask& availableLayerClusters) const override {
    auto filteredLayerClusters = std::make_unique<HgcalClusterFilterMask>();
    for (auto const& cl : availableLayerClusters) {
      if (layerClusters[cl.first].algo() == algo_number_ ||
          layerClusters[cl.first].hitsAndFractions().size() <= max_cluster_size_)
        filteredLayerClusters->emplace_back(cl);
    }
    return filteredLayerClusters;
  }

 private:
  int algo_number_;
  unsigned int max_cluster_size_;
};
}

#endif
