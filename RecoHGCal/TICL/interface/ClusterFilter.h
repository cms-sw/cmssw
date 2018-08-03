// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef RecoHGCal_TICL_ClusterFilter_H__
#define RecoHGCal_TICL_ClusterFilter_H__

#include "RecoHGCal/TICL/interface/ClusterFilterBase.h"

#include <memory>

class ClusterFilter {
public:
  ClusterFilter() {}
  explicit ClusterFilter(std::unique_ptr<ClusterFilterBase> ofilter): myfilter_(std::move(ofilter)) {}

  void swap(ClusterFilter& o) { std::swap(myfilter_, o.myfilter_); }

  std::unique_ptr<std::vector<std::pair<unsigned int, float> > > filter(const std::vector<reco::CaloCluster>& layerClusters,
                const std::vector<std::pair<unsigned int, float> >& mask) const { return myfilter_->filter(layerClusters, mask); }

private:
  std::unique_ptr<ClusterFilterBase> myfilter_;
};

#endif
