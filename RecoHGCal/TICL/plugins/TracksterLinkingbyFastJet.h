#ifndef RecoHGCal_TICL_TracksterLinkingAlgoByFastJet_H
#define RecoHGCal_TICL_TracksterLinkingAlgoByFastJet_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"
#include "fastjet/ClusterSequence.hh"
#include "DataFormats/Math/interface/deltaR.h"

namespace ticl {

  class TracksterLinkingbyFastJet : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbyFastJet(const edm::ParameterSet& conf,
                              edm::ConsumesCollector iC,
                              cms::Ort::ONNXRuntime const* onnxRuntime = nullptr)
        : TracksterLinkingAlgoBase(conf, iC), radius_(conf.getParameter<double>("radius")) {
      // Cluster tracksters into jets using FastJet with configurable algorithm
      auto algo = conf.getParameter<int>("jet_algorithm");

      switch (algo) {
        case 0:
          algorithm_ = fastjet::kt_algorithm;
          break;
        case 1:
          algorithm_ = fastjet::cambridge_algorithm;
          break;
        case 2:
          algorithm_ = fastjet::antikt_algorithm;
          break;
        default:
          throw cms::Exception("BadConfig") << "FastJet jet clustering algorithm not set correctly.";
      }
    }

    ~TracksterLinkingbyFastJet() override {}

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;

    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override{};

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<int>("algo_verbosity", 0);
      iDesc.add<int>("jet_algorithm", 2)
          ->setComment("FastJet jet clustering algorithm: 0 = kt, 1 = Cambridge/Aachen, 2 = anti-kt");
      iDesc.add<double>("radius", 0.1);
    }

  private:
    fastjet::JetAlgorithm algorithm_;  // FastJet jet clustering algorithm
    const float radius_;
  };

}  // namespace ticl

#endif
