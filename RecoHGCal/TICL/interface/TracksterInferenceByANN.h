#ifndef RecoHGCal_TICL_TracksterInferenceByANN_H__
#define RecoHGCal_TICL_TracksterInferenceByANN_H__

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"

namespace ticl {
  class TracksterInferenceByANN : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByANN(const edm::ParameterSet& conf);
    void inputData(const std::vector<reco::CaloCluster>& layerClusters, std::vector<Trackster>& tracksters) override;
    void runInference(std::vector<Trackster>& tracksters) override;

  private:
    const cms::Ort::ONNXRuntime* onnxPIDSession_;
    const cms::Ort::ONNXRuntime* onnxEnergySession_;
  };
}  // namespace ticl

#endif
