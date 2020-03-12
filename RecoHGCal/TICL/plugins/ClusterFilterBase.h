// Author: Felice Pantaleo, Marco Rovere - felice.pantaleo@cern.ch, marco.rovere@cern.ch
// Date: 09/2018

#ifndef RecoHGCal_TICL_ClusterFilterBase_H__
#define RecoHGCal_TICL_ClusterFilterBase_H__

#include "DataFormats/HGCalReco/interface/Common.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
}
namespace reco {
  class CaloCluster;
}

namespace ticl {
  class ClusterFilterBase {
  public:
    explicit ClusterFilterBase(const edm::ParameterSet&){};
    virtual ~ClusterFilterBase(){};

    virtual std::unique_ptr<HgcalClusterFilterMask> filter(const std::vector<reco::CaloCluster>& layerClusters,
                                                           const HgcalClusterFilterMask& mask,
                                                           std::vector<float>& layerClustersMask,
                                                           hgcal::RecHitTools& rhtools) const = 0;
  };
}  // namespace ticl

#endif
