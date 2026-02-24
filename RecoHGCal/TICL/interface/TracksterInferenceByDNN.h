#ifndef RecoHGCal_TICL_TracksterInferenceByDNN_H__
#define RecoHGCal_TICL_TracksterInferenceByDNN_H__

#include <string>
#include <vector>

#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"
#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

// TracksterInferenceByDNN.h

namespace ticl {

  class TracksterInferenceByDNN final : public TracksterInferenceAlgoBase {
  public:
    explicit TracksterInferenceByDNN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);

    void runInference(const std::vector<reco::CaloCluster>& layerClusters,
                      std::vector<Trackster>& tracksters,
                      const hgcal::RecHitTools& rhtools) const override;

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  private:
    // Sessions are owned by the GlobalCache.
    cms::Ort::ONNXRuntime const* onnxPIDSession_ = nullptr;
    cms::Ort::ONNXRuntime const* onnxEnergySession_ = nullptr;

    const std::vector<std::string> inputNames_;
    const std::vector<std::string> output_en_;
    const std::vector<std::string> output_id_;

    const float eidMinClusterEnergy_;
    const int eidNLayers_;
    const int eidNClusters_;
    static constexpr int eidNFeatures_ = 3;

    const int doPID_;
    const int doRegression_;
    const int miniBatchSize_;
    bool enabled_ = false;
  };

}  // namespace ticl

#endif  // RecoHGCal_TICL_TracksterInferenceByDNN_H__
