#ifndef RecoHGCal_TICL_TracksterInferenceByPFN_H__
#define RecoHGCal_TICL_TracksterInferenceByPFN_H__

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"

namespace ticl {

  class TracksterInferenceByPFN : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByPFN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);
    void inputData(const std::vector<reco::CaloCluster>& layerClusters,
                   std::vector<Trackster>& tracksters,
                   const hgcal::RecHitTools& rhtools) override;
    void runInference(std::vector<Trackster>& tracksters) override;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    const cms::Ort::ONNXRuntime* onnxPIDSession_;
    const cms::Ort::ONNXRuntime* onnxEnergySession_;

    const std::vector<std::string> inputNames_;
    const std::vector<std::string> output_en_;
    const std::vector<std::string> output_id_;
    const float eidMinClusterEnergy_;
    const int eidNLayers_;
    const int eidNClusters_;
    static constexpr int eidNFeatures_ = 7;
    int doPID_;
    int doRegression_;
    bool enabled_ = false;

    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<int> tracksterIndices_;
    std::vector<std::vector<float>> input_Data_;
    int batchSize_;
  };
}  // namespace ticl

#endif  // RecoHGCal_TICL_TracksterInferenceByPFN_H__
