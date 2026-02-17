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

// TracksterInferenceAlgoBase.h

namespace ticl {

  class TracksterInferenceAlgoBase {
  public:
    struct OrtScratch {
      // Reused buffers for ONNXRuntime::runInto()
      cms::Ort::FloatArrays inputs;
      cms::Ort::FloatArrays outputs;

      // Reused shapes vector: one entry per input tensor.
      std::vector<std::vector<int64_t>> input_shapes;

      void clearPerEvent() {
        // Keep capacity, reset sizes.
        outputs.clear();
        // Do NOT clear inputs or input_shapes here: we want to reuse capacity and overwrite per minibatch.
      }
    };

    explicit TracksterInferenceAlgoBase(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")), cache_(cache) {}

    virtual ~TracksterInferenceAlgoBase() = default;

    // New API: build minibatches internally
    virtual void runInference(const std::vector<reco::CaloCluster>& layerClusters,
                              std::vector<Trackster>& tracksters,
                              const hgcal::RecHitTools& rhtools) const = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); }

  protected:
    int algo_verbosity_;
    TICLONNXGlobalCache const* cache_;

    // Per-stream scratch: must be mutable because runInference() is const.
    mutable OrtScratch ortScratch_;
  };

}  // namespace ticl

#endif
