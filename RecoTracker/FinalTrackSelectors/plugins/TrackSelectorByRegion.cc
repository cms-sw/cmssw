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
      : tracksToken_(consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("tracks"))) {
    inputTrkRegionToken_ = consumes<edm::OwnVector<TrackingRegion>>(conf.getParameter<edm::InputTag>("regions"));
    produce_collection = conf.getParameter<bool>("produceTrackCollection");
    produce_mask = conf.getParameter<bool>("produceMask");
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
    if (not produce_collection)
      if (not produce_mask)
        return;
    // products
    auto mask = std::make_unique<MaskCollection>();                  // mask w/ the same size of the input collection
    auto output_tracks = std::make_unique<reco::TrackCollection>();  // selected output collection

    auto regionsHandle = iEvent.getHandle(inputTrkRegionToken_);
    auto tracksHandle = iEvent.getHandle(tracksToken_);

    const auto& tracks = *tracksHandle;
    mask->assign(tracks.size(), false);
    const auto& regions = *regionsHandle;

    for (const auto& region : regions)
      if (const auto* roi = dynamic_cast<const TrackingRegion*>(&region)) {
        auto amask = roi->checkTracks(tracks);
        for (size_t it = 0; it < amask.size(); it++) {
          mask->at(it) = mask->at(it) or amask.at(it);
        }
      }

    if (produce_collection)
      for (size_t it = 0; it < mask->size(); it++)
        if (mask->at(it))
          output_tracks->push_back(tracks[it]);

    if (produce_mask)
      iEvent.put(std::move(mask));
    if (produce_collection)
      iEvent.put(std::move(output_tracks));
  }
  bool produce_collection;
  bool produce_mask;
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion>> inputTrkRegionToken_;
};

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackSelectorByRegion);
