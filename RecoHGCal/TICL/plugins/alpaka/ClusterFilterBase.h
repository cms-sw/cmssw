
#pragma once

#include "DataFormats/HGCalReco/interface/Common.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include <memory>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class ClusterFilterBase {
  public:
    explicit ClusterFilterBase(const edm::ParameterSet&) = default;
    virtual ~ClusterFilterBase() = default;

    virtual void filter(const HGCalSoAClustersDeviceCollection& layerClusters,
                        std::vector<float>& layerClustersMask,
                        hgcal::RecHitTools& rhtools) const = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
