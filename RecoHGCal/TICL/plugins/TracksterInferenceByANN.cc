#include "RecoHGCal/TICL/interface/TracksterInferenceByANN.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ticl {

  TracksterInferenceByANN::TracksterInferenceByANN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache) {
    // Read model paths as strings. Empty string means disabled.
    onnxPIDModelPath_ = conf.getParameter<std::string>("onnxPIDModelPath");
    onnxEnergyModelPath_ = conf.getParameter<std::string>("onnxEnergyModelPath");

    // Resolve sessions from the cache only if paths are non-empty.
    if (cache_ != nullptr) {
      if (!onnxPIDModelPath_.empty()) {
        onnxPIDSession_ = cache_->getByModelPathString(onnxPIDModelPath_);
      }
      if (!onnxEnergyModelPath_.empty()) {
        onnxEnergySession_ = cache_->getByModelPathString(onnxEnergyModelPath_);
      }
    }

    // Enable if at least one session is available.
    enabled_ = (onnxPIDSession_ != nullptr) || (onnxEnergySession_ != nullptr);
  }

  void TracksterInferenceByANN::inputData(const std::vector<reco::CaloCluster>& layerClusters,
                                          std::vector<Trackster>& tracksters,
                                          const hgcal::RecHitTools& rhtools) {
    if (!enabled_) {
      return;
    }
  }

  void TracksterInferenceByANN::runInference(std::vector<Trackster>& tracksters) {
    if (!enabled_) {
      return;
    }
  }

  void TracksterInferenceByANN::fillPSetDescription(edm::ParameterSetDescription& desc) {
    TracksterInferenceAlgoBase::fillPSetDescription(desc);

    desc.add<std::string>("onnxPIDModelPath", "")
        ->setComment("Path to ONNX PID model. If empty, PID inference is skipped.");
    desc.add<std::string>("onnxEnergyModelPath", "")
        ->setComment("Path to ONNX energy model. If empty, energy regression is skipped.");

    // Add your ANN-specific configuration here as you implement it.
  }

}  // namespace ticl
