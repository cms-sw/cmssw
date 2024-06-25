// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 11/2018

#ifndef RecoHGCal_TICL_ClusterFilterByAlgo_H__
#define RecoHGCal_TICL_ClusterFilterByAlgo_H__

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "ClusterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl {
  class ClusterFilterByAlgo final : public ClusterFilterBase {
  public:
    ClusterFilterByAlgo(const edm::ParameterSet& ps)
        : ClusterFilterBase(ps), algo_number_(ps.getParameter<std::vector<int>>("algo_number")) {}
    ~ClusterFilterByAlgo() override{};

    void filter(const std::vector<reco::CaloCluster>& layerClusters,
                std::vector<float>& layerClustersMask,
                hgcal::RecHitTools& rhtools) const override {
      for (size_t i = 0; i < layerClusters.size(); i++) {
        if (find(algo_number_.begin(), algo_number_.end(), layerClusters[i].algo()) == algo_number_.end()) {
          layerClustersMask[i] = 0.;
        }
      }
    }

  private:
    std::vector<int> algo_number_;
  };
}  // namespace ticl

#endif
