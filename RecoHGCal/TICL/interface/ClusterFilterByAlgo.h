// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef RecoHGCal_TICL_ClusterFilterByAlgo_H__
#define RecoHGCal_TICL_ClusterFilterByAlgo_H__

#include "RecoHGCal/TICL/interface/ClusterFilterBase.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
class ClusterFilterByAlgo final : public ClusterFilterBase {
public:
  ClusterFilterByAlgo(const edm::ParameterSet & ps)
    : ClusterFilterBase(ps),
  algo_number_(ps.getParameter<int>("algo_number")){}

  std::unique_ptr<std::vector<std::pair<unsigned int, float> > >
    filter(const std::vector<reco::CaloCluster>& layerClusters,
           const std::vector<std::pair<unsigned int, float> >& availableLayerClusters) const override
    {
      auto filteredLayerClusters = std::make_unique<std::vector<std::pair<unsigned int, float>>>();
      for (auto const & cl : availableLayerClusters) {
        if (layerClusters[cl.first].algo() == algo_number_)
          filteredLayerClusters->emplace_back(cl);
      }
      return filteredLayerClusters;
    }
private:
  int algo_number_;
};

#endif
