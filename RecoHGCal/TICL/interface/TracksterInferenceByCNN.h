#ifndef RecoHGCal_TICL_interface_TracksterInferenceByCNN_h
#define RecoHGCal_TICL_interface_TracksterInferenceByCNN_h

#include "RecoHGCal/TICL/interface/TICLONNXGlobalCache.h"
#include "RecoHGCal/TICL/interface/TracksterInferenceAlgoBase.h"

#include <string>
#include <vector>

namespace reco {
  class CaloCluster;
}

namespace hgcal {
  class RecHitTools;
}

namespace ticl {

  class TracksterInferenceByCNN : public TracksterInferenceAlgoBase {
  public:
    TracksterInferenceByCNN(const edm::ParameterSet& conf, TICLONNXGlobalCache const* cache);

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

    void runInference(const std::vector<reco::CaloCluster>& layerClusters,
                      std::vector<Trackster>& tracksters,
                      const hgcal::RecHitTools& rhtools) const override;

  private:
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;

    double eidMinClusterEnergy_;
    int eidNLayers_;
    int eidNClusters_;
    int doPID_;
    int miniBatchSize_;

cms::Ort::ONNXRuntime const* onnxSession_ = nullptr;
    bool enabled_ = false;

    static constexpr int eidNFeatures_ = 3;
  };

}  // namespace ticl

#endif