#ifndef RecoHGCal_TICL_TICLInterpretationAlgoBase_h
#define RecoHGCal_TICL_TICLInterpretationAlgoBase_h

#include <memory>
#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm
namespace ticl {
  // Sentinel a muon interpretation pass writes into resultCandidate for a track it
  // rejected (the trajectory points to a shower, so it is not a muon). The producer
  // routes such tracks to the next (general) pass instead of building a muon candidate.
  constexpr int kMuonRejected = -2;

  template <typename T>
  class TICLInterpretationAlgoBase {
  public:
    TICLInterpretationAlgoBase(const edm::ParameterSet& conf, edm::ConsumesCollector)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}
    virtual ~TICLInterpretationAlgoBase() {}
    struct Inputs {
      const edm::Event& ev;
      const edm::EventSetup& es;
      const std::vector<reco::CaloCluster>& layerClusters;
      const edm::ValueMap<std::pair<float, float>>& layerClustersTime;
      const edm::MultiSpan<Trackster>& tracksters;
      const std::vector<std::vector<unsigned int>>& linkedResultTracksters;
      const edm::Handle<std::vector<T>> tracksHandle;
      const std::vector<bool>& maskedTracks;

      Inputs(const edm::Event& eV,
             const edm::EventSetup& eS,
             const std::vector<reco::CaloCluster>& lC,
             const edm::ValueMap<std::pair<float, float>>& lcT,
             const edm::MultiSpan<Trackster>& tS,
             const std::vector<std::vector<unsigned int>>& links,
             const edm::Handle<std::vector<T>> trks,
             const std::vector<bool>& mT)
          : ev(eV),
            es(eS),
            layerClusters(lC),
            layerClustersTime(lcT),
            tracksters(tS),
            linkedResultTracksters(links),
            tracksHandle(trks),
            maskedTracks(mT) {}
    };

    struct TrackTimingInformation {
      const edm::Handle<edm::ValueMap<float>> tkTime_h;
      const edm::Handle<edm::ValueMap<float>> tkTimeErr_h;
      const edm::Handle<edm::ValueMap<float>> tkQuality_h;
      const edm::Handle<edm::ValueMap<float>> tkBeta_h;
      const edm::Handle<edm::ValueMap<float>> tkPath_h;
      const edm::Handle<edm::ValueMap<GlobalPoint>> tkMtdPos_h;

      TrackTimingInformation(const edm::Handle<edm::ValueMap<float>> tkT,
                             const edm::Handle<edm::ValueMap<float>> tkTE,
                             const edm::Handle<edm::ValueMap<float>> tkQ,
                             const edm::Handle<edm::ValueMap<float>> tkB,
                             const edm::Handle<edm::ValueMap<float>> tkP,
                             const edm::Handle<edm::ValueMap<GlobalPoint>> mtdPos)
          : tkTime_h(tkT), tkTimeErr_h(tkTE), tkQuality_h(tkQ), tkBeta_h(tkB), tkPath_h(tkP), tkMtdPos_h(mtdPos) {}
    };

    // maskedTracksters (indexed over input.tracksters) lets several interpretation
    // passes run in sequence: a pass marks the tracksters it consumes, and later
    // passes skip them. Empty or all-false means "nothing already consumed".
    virtual void makeCandidates(const Inputs& input,
                                edm::Handle<MtdHostCollection> inputTiming_h,
                                std::vector<Trackster>& resultTracksters,
                                std::vector<int>& resultCandidate,
                                std::vector<bool>& maskedTracksters) = 0;

    virtual void initialize(const HGCalDDDConstants* hgcons,
                            const hgcal::RecHitTools rhtools,
                            const edm::ESHandle<MagneticField> bfieldH,
                            const edm::ESHandle<Propagator> propH) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); };

  protected:
    int algo_verbosity_;
  };
}  // namespace ticl

#endif
