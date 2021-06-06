#ifndef RecoTracker_FinalTrackSelectors_TrackMVAClassifierBase_h
#define RecoTracker_FinalTrackSelectors_TrackMVAClassifierBase_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/GBRForest/interface/GBRForest.h"

#include <vector>
#include <memory>

class TrackMVAClassifierBase : public edm::stream::EDProducer<> {
public:
  explicit TrackMVAClassifierBase(const edm::ParameterSet& cfg);
  ~TrackMVAClassifierBase() override;

  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;

  //Collection with pairs <MVAOutput, isReliable>
  using MVAPairCollection = std::vector<std::pair<float, bool>>;

protected:
  static void fill(edm::ParameterSetDescription& desc);

  virtual void initEvent(const edm::EventSetup& es) = 0;

  virtual void computeMVA(reco::TrackCollection const& tracks,
                          reco::BeamSpot const& beamSpot,
                          reco::VertexCollection const& vertices,
                          MVAPairCollection& mvas) const = 0;

private:
  void produce(edm::Event& evt, const edm::EventSetup& es) final;

  /// source collection label
  edm::EDGetTokenT<reco::TrackCollection> src_;
  edm::EDGetTokenT<reco::BeamSpot> beamspot_;
  edm::EDGetTokenT<reco::VertexCollection> vertices_;

  bool ignoreVertices_;

  // MVA

  // qualitycuts (loose, tight, hp)
  float qualityCuts[3];
};

namespace trackMVAClassifierImpl {
  template <typename EventCache>
  struct ComputeMVA {
    template <typename MVA>
    void operator()(MVA const& mva,
                    reco::TrackCollection const& tracks,
                    reco::BeamSpot const& beamSpot,
                    reco::VertexCollection const& vertices,
                    TrackMVAClassifierBase::MVAPairCollection& mvas) {
      EventCache cache;

      size_t current = 0;
      for (auto const& trk : tracks) {
        mvas[current++] = mva(trk, beamSpot, vertices, cache);
      }
    }
  };

  template <>
  struct ComputeMVA<void> {
    template <typename MVA>
    void operator()(MVA const& mva,
                    reco::TrackCollection const& tracks,
                    reco::BeamSpot const& beamSpot,
                    reco::VertexCollection const& vertices,
                    TrackMVAClassifierBase::MVAPairCollection& mvas) {
      size_t current = 0;
      for (auto const& trk : tracks) {
        //BDT outputs are considered always reliable. Hence "true"
        std::pair<float, bool> output(mva(trk, beamSpot, vertices), true);
        mvas[current++] = output;
      }
    }
  };
}  // namespace trackMVAClassifierImpl

template <typename MVA, typename EventCache = void>
class TrackMVAClassifier : public TrackMVAClassifierBase {
public:
  explicit TrackMVAClassifier(const edm::ParameterSet& cfg)
      : TrackMVAClassifierBase(cfg), mva(cfg.getParameter<edm::ParameterSet>("mva")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    fill(desc);
    edm::ParameterSetDescription mvaDesc;
    MVA::fillDescriptions(mvaDesc);
    desc.add<edm::ParameterSetDescription>("mva", mvaDesc);
    descriptions.add(MVA::name(), desc);
  }

private:
  void beginStream(edm::StreamID) final { mva.beginStream(); }

  void initEvent(const edm::EventSetup& es) final { mva.initEvent(es); }

  void computeMVA(reco::TrackCollection const& tracks,
                  reco::BeamSpot const& beamSpot,
                  reco::VertexCollection const& vertices,
                  MVAPairCollection& mvas) const final {
    trackMVAClassifierImpl::ComputeMVA<EventCache> computer;
    computer(mva, tracks, beamSpot, vertices, mvas);
  }

  MVA mva;
};

#endif  //  RecoTracker_FinalTrackSelectors_TrackMVAClassifierBase_h
