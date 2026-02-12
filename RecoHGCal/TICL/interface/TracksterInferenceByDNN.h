#ifndef RecoHGCal_TICL_TracksterInferenceByDNN_H__
#define RecoHGCal_TICL_TracksterInferenceByDNN_H__

#include <string>
#include <vector>

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace ticl {

  class TracksterInferenceByDNN : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByDNN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);

    void inputData(const std::vector<reco::CaloCluster>& layerClusters,
                   std::vector<Trackster>& tracksters,
                   const hgcal::RecHitTools& rhtools) override;

    void runInference(std::vector<Trackster>& tracksters) override;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    // Sessions are owned by the GlobalCache.
    cms::Ort::ONNXRuntime const* onnxPIDSession_ = nullptr;
    cms::Ort::ONNXRuntime const* onnxEnergySession_ = nullptr;

    // Reuse input and output buffers
    cms::Ort::FloatArrays inputs_;
    cms::Ort::FloatArrays outputs_;
    const std::vector<std::string> inputNames_;
    const std::vector<std::string> output_en_;
    const std::vector<std::string> output_id_;
    const float eidMinClusterEnergy_;
    const int eidNLayers_;
    const int eidNClusters_;
    static constexpr int eidNFeatures_ = 3;
    const int doPID_;
    const int doRegression_;

    // True if at least one required model is available.
    bool enabled_ = false;

    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<int> tracksterIndices_;
    std::vector<std::vector<float>> input_Data_;
    int batchSize_ = 0;
  };

}  // namespace ticl

#endif  // RecoHGCal_TICL_TracksterInferenceByDNN_H__
