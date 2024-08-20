#ifndef RecoHGCal_TICL_TracksterLinkingPassthrough_H
#define RecoHGCal_TICL_TracksterLinkingPassthrough_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"

namespace ticl {

  class TracksterLinkingPassthrough : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingPassthrough(const edm::ParameterSet& conf,
                                edm::ConsumesCollector iC,
                                cms::Ort::ONNXRuntime const* onnxRuntime = nullptr)
        : TracksterLinkingAlgoBase(conf, iC, onnxRuntime) {}

    ~TracksterLinkingPassthrough() override {}

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;
    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override{};

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) { iDesc.add<int>("algo_verbosity", 0); }
  };

}  // namespace ticl

#endif
