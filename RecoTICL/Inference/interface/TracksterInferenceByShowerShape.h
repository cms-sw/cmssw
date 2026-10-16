#ifndef RecoTICL_Inference_TracksterInferenceByShowerShape_h
#define RecoTICL_Inference_TracksterInferenceByShowerShape_h

// Identification of tracksters from their shower shape, with one ONNX model. Mode "em": 16 trackster quantities give
// the EM score s; the trackster gets P(photon) = s and P(charged hadron) = 1 - s. Mode "mixture": the 16 quantities and
// the 6 values of the layer-cluster profile give the 8 probabilities in ParticleType order.

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "RecoTICL/Inference/interface/TICLONNXGlobalCache.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoBase.h"

namespace ticl {

  class TracksterInferenceByShowerShape : public TracksterInferenceAlgoBase {
  public:
    static constexpr std::size_t kNFeatures = 16;
    static constexpr std::size_t kNMixtureFeatures = kNFeatures + 6;

    TracksterInferenceByShowerShape(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void runInference(const std::vector<reco::CaloCluster>& layerClusters,
                      std::vector<Trackster>& tracksters,
                      const ticlgeom::Tools& rhtools) const override;

    // log(1 + raw energy), CE-E energy over raw energy, |eta| and |z| of the barycenter, log(1 + number of layer
    // clusters), the logs of the three PCA eigenvalues, the first eigenvalue over their sum, the three sigmas, the three
    // PCA sigmas, and |eta of the main axis - eta of the barycenter|. A zero main axis has eta 0, so the last quantity
    // is then |eta| of the barycenter. The features use eigenvalues of at least 1e-6.
    static std::array<float, kNFeatures> features(const Trackster& trackster);

  private:
    bool mixture_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    float minEnergy_;
    int doPID_;
    cms::Ort::ONNXRuntime const* onnxSession_ = nullptr;
  };

}  // namespace ticl

#endif
