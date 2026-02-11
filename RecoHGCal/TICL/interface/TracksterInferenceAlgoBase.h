#ifndef RecoHGCal_TICL_TracksterInferenceAlgo_H__
#define RecoHGCal_TICL_TracksterInferenceAlgo_H__

#include <vector>
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {

  class TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceAlgoBase(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")), cache_(cache) {}

    virtual ~TracksterInferenceAlgoBase() {}

    virtual void inputData(const std::vector<reco::CaloCluster>& layerClusters,
                           std::vector<Trackster>& tracksters,
                           const hgcal::RecHitTools& rhtools) = 0;

    virtual void runInference(std::vector<Trackster>& tracksters) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); }

  protected:
    int algo_verbosity_;
    TICLONNXGlobalCache const* cache_;
  };

}  // namespace ticl

#endif
