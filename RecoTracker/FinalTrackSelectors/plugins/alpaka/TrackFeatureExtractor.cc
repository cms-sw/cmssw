#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackTorchClassifierFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackFeatureExtractor : public stream::EDProducer<> {
  public:
    TrackFeatureExtractor(const edm::ParameterSet& iConfig)
        : EDProducer<>(iConfig),
          tracksInput_token_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
          beamspot_token_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
          featuresPut_token_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTracks"));
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      auto const& tracks = iEvent.get(tracksInput_token_);
      auto const& beamspot = iEvent.get(beamspot_token_);

      const auto nTracks = tracks.size();

      // Create HOST collection first, fill it, then copy to device
      PortableHostCollection<TrackTorchClassifierFeaturesSoA> features_host(nTracks);

      auto features_view = features_host.view();

      for (size_t i = 0; i < nTracks; ++i) {
        const auto& track = tracks[i];

        features_view[i].dxyBeamSpot() = track.dxy(beamspot.position());
        features_view[i].dzBeamSpot() = track.dz(beamspot.position());
        features_view[i].dxyError() = track.dxyError();
        features_view[i].dzError() = track.dzError();

        features_view[i].normalizedChi2() = track.normalizedChi2();
        features_view[i].eta() = track.eta();
        features_view[i].phi() = track.phi();
        features_view[i].etaError() = track.etaError();
        features_view[i].phiError() = track.phiError();
        features_view[i].ndof() = track.ndof();

        const auto& hitPattern = track.hitPattern();
        features_view[i].lostInnerHits() = hitPattern.numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
        features_view[i].lostOuterHits() = hitPattern.numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
        features_view[i].layersWithoutMeas() = hitPattern.trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
        features_view[i].validPixelHits() = hitPattern.numberOfValidPixelHits();
        features_view[i].validStripHits() = hitPattern.numberOfValidStripHits();
      }

      // Create device collection and copy from host
      TrackFeaturesDeviceCollection features_device(iEvent.queue(), nTracks);
      alpaka::memcpy(iEvent.queue(), features_device.buffer(), features_host.const_buffer());

      iEvent.emplace(featuresPut_token_, std::move(features_device));
    }

  private:
    const edm::EDGetTokenT<reco::TrackCollection> tracksInput_token_;
    const edm::EDGetTokenT<reco::BeamSpot> beamspot_token_;
    const device::EDPutToken<TrackFeaturesDeviceCollection> featuresPut_token_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackFeatureExtractor);
