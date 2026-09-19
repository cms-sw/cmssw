#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackSoA/interface/PixelTrackSoATab.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"

class HLTPixelTrackSoATableProducer : public edm::global::EDProducer<> {
  using TrackSoAHost = reco::TracksHost;

public:
  explicit HLTPixelTrackSoATableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<TrackSoAHost> trackSoAToken_;

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override;
};

HLTPixelTrackSoATableProducer::HLTPixelTrackSoATableProducer(const edm::ParameterSet& iConfig)
    : trackSoAToken_(consumes<TrackSoAHost>(iConfig.getParameter<edm::InputTag>("trackSrc"))) {
  produces<std::vector<PixelTrackSoATab>>();
}

void HLTPixelTrackSoATableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  const auto& tracksHost = iEvent.get(trackSoAToken_);
  const auto view = tracksHost.const_view();

  auto out = std::make_unique<std::vector<PixelTrackSoATab>>();
  out->reserve(view.tracks().nTracks());

  for (int i = 0; i < view.tracks().nTracks(); ++i)
    out->emplace_back(view.tracks(), i);

  iEvent.put(std::move(out));
}

void HLTPixelTrackSoATableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("pixelTracksAlpaka"));

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPixelTrackSoATableProducer);
