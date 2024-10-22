#ifndef RecoHGCal_TICL_TracksterInferenceByCNNv4_H__
#define RecoHGCal_TICL_TracksterInferenceByCNNv4_H__

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ticl {

  class TracksterInferenceByCNNv4 : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByCNNv4(const edm::ParameterSet& conf);
    void inputData(const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& tracksters) override;
    void runInference(std::vector<Trackster>& tracksters) override;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    const cms::Ort::ONNXRuntime* onnxSession_;

    const std::string modelPath_;
    const std::vector<std::string> inputNames_;
    const std::vector<std::string> outputNames_;
    const float eidMinClusterEnergy_;
    const int eidNLayers_;
    const int eidNClusters_;
    static constexpr int eidNFeatures_ = 3;
    int doPID_;
    int doRegression_;

    hgcal::RecHitTools rhtools_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<int> tracksterIndices_;
    std::vector<std::vector<float>> input_Data_;
    int batchSize_;
  };
}  // namespace ticl

#endif  // RecoHGCal_TICL_TracksterInferenceByDNN_H__
