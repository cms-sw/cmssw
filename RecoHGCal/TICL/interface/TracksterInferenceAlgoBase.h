#ifndef RecoHGCal_TICL_TracksterInferenceAlgo_H__
#define RecoHGCal_TICL_TracksterInferenceAlgo_H__

#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {

  class TracksterInferenceAlgoBase {
  public:
    struct OrtScratch {
      // Used with ONNXRuntime::runInto() to reuse buffers across events.
      cms::Ort::FloatArrays inputs;
      cms::Ort::FloatArrays outputs;

      void clearPerEvent() {
        // Keep capacity, reset sizes.
        inputs.clear();
        outputs.clear();
      }
    };

    explicit TracksterInferenceAlgoBase(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")), cache_(cache) {}

    virtual ~TracksterInferenceAlgoBase() = default;

    virtual void inputData(const std::vector<reco::CaloCluster>& layerClusters,
                           std::vector<Trackster>& tracksters,
                           const hgcal::RecHitTools& rhtools) = 0;

    virtual void runInference(std::vector<Trackster>& tracksters) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); }

  protected:
    int algo_verbosity_;
    TICLONNXGlobalCache const* cache_;

    // Per-stream scratch (safe because each stream has its own module + plugin instances).
    mutable OrtScratch ortScratch_;
  };

}  // namespace ticl

#endif
