#ifndef RecoHGCal_TICL_TracksterInferenceByANN_H__
#define RecoHGCal_TICL_TracksterInferenceByANN_H__

#include <string>

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"

namespace ticl {

  class TracksterInferenceByANN : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByANN(const edm::ParameterSet&, TICLONNXGlobalCache const* cache);

    void inputData(const std::vector<reco::CaloCluster>& layerClusters,
                   std::vector<Trackster>& tracksters,
                   const hgcal::RecHitTools& rhtools) override;

    void runInference(std::vector<Trackster>& tracksters) override;

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

  private:
    // Sessions are owned by the GlobalCache, this class only holds raw pointers.
    cms::Ort::ONNXRuntime const* onnxPIDSession_ = nullptr;
    cms::Ort::ONNXRuntime const* onnxEnergySession_ = nullptr;

    // If false, inputData/runInference become no-ops.
    bool enabled_ = false;

    // Keep the configured model path strings (useful for debugging if needed).
    std::string onnxPIDModelPath_;
    std::string onnxEnergyModelPath_;

    // Your ANN input/output configuration can go here as you implement it.
  };

}  // namespace ticl

#endif
