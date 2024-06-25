// Author: Felice Pantaleo, Wahid Redjeb, Aurora Perego (CERN) - felice.pantaleo@cern.ch, wahid.redjeb@cern.ch, aurora.perego@cern.ch
// Date: 12/2023
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

using namespace ticl;

class MergedTrackstersProducer : public edm::stream::EDProducer<> {
public:
  explicit MergedTrackstersProducer(const edm::ParameterSet &ps);
  ~MergedTrackstersProducer() override{};
  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  edm::EDGetTokenT<std::vector<Trackster>> egamma_tracksters_token_;

  edm::EDGetTokenT<std::vector<Trackster>> general_tracksters_token_;
};

MergedTrackstersProducer::MergedTrackstersProducer(const edm::ParameterSet &ps)
    : egamma_tracksters_token_(
          consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("egamma_tracksters"))),
      general_tracksters_token_(
          consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("had_tracksters"))) {
  produces<std::vector<Trackster>>();
}

void MergedTrackstersProducer::produce(edm::Event &evt, const edm::EventSetup &es) {
  auto resultTracksters = std::make_unique<std::vector<Trackster>>();
  auto const &egamma_tracksters = evt.get(egamma_tracksters_token_);
  auto const &had_tracksters = evt.get(general_tracksters_token_);
  for (auto const &eg_trackster : egamma_tracksters) {
    resultTracksters->push_back(eg_trackster);
  }
  for (auto const &had_trackster : had_tracksters) {
    resultTracksters->push_back(had_trackster);
  }

  evt.put(std::move(resultTracksters));
}

void MergedTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("egamma_tracksters", edm::InputTag("ticlTrackstersCLUE3DEM"));
  desc.add<edm::InputTag>("had_tracksters", edm::InputTag("ticlTrackstersCLUE3DHAD"));
  descriptions.add("mergedTrackstersProducer", desc);
}

DEFINE_FWK_MODULE(MergedTrackstersProducer);
