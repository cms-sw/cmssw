#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include <vector>
#include <memory>

class TrackSelectorByRegion final : public edm::global::EDProducer<> {
public:
  explicit TrackSelectorByRegion(const edm::ParameterSet& conf)
      : produce_collection(conf.getParameter<bool>("produceTrackCollection")),
        produce_mask(conf.getParameter<bool>("produceMask")),
        tracksToken_(consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks"))),
        inputTrkRegionToken_(consumes<edm::OwnVector<TrackingRegion>>(conf.getParameter<edm::InputTag>("regions"))) {
    if (produce_collection)
      produces<reco::TrackCollection>();
    if (produce_mask)
      produces<std::vector<bool>>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracks", edm::InputTag("hltPixelTracks"));
    desc.add<edm::InputTag>("regions", edm::InputTag(""));
    desc.add<bool>("produceTrackCollection", true);
    desc.add<bool>("produceMask", true);
    descriptions.add("trackSelectorByRegion", desc);
  }

private:
  using MaskCollection = std::vector<bool>;

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override {
    if (not produce_collection and not produce_mask)
      return;

    auto regionsHandle = iEvent.getHandle(inputTrkRegionToken_);
    auto tracksHandle = iEvent.getHandle(tracksToken_);

    const auto& tracks = *tracksHandle;
    auto mask = std::make_unique<MaskCollection>(tracks.size(), false);  // output mask
    const auto& regions = *regionsHandle;

    for (auto const& region : regions) {
      auto const& region_mask = region.checkTracks(tracks);
      for (size_t i = 0; i < region_mask.size(); ++i) {
        mask->at(i) = mask->at(i) or region_mask[i];
      }
    }

    if (produce_collection) {
      auto output_tracks = std::make_unique<reco::TrackCollection>();  // selected output collection
      size_t size = 0;
      for (size_t i = 0; i < mask->size(); i++) {
        if (mask->at(i))
          ++size;
      }
      output_tracks->reserve(size);
      for (size_t i = 0; i < mask->size(); i++) {
        if (mask->at(i))
          output_tracks->push_back(tracks[i]);
      }
      iEvent.put(std::move(output_tracks));
    }
    if (produce_mask) {
      iEvent.put(std::move(mask));
    }
  }

  const bool produce_collection;
  const bool produce_mask;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<edm::OwnVector<TrackingRegion>> inputTrkRegionToken_;
};

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackSelectorByRegion);
