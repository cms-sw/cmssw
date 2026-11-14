// Author: Mohamed Darwish
// TrackMuonTableProducer.cc

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

class TrackMuonTableProducer : public edm::global::EDProducer<> {
public:
  explicit TrackMuonTableProducer(const edm::ParameterSet& cfg)
      : tracksToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
        muonsToken_(consumes<reco::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        name_(cfg.getParameter<std::string>("name")),
        doc_(cfg.getParameter<std::string>("doc")) {
    produces<nanoaod::FlatTable>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
    desc.add<edm::InputTag>("muons", edm::InputTag("muons1stStep"));
    desc.add<std::string>("name", "TrackMuon");
    desc.add<std::string>("doc", "Muons matched to GeneralTrack, one row per muon");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override {
    edm::Handle<std::vector<reco::Track>> tracksH;
    evt.getByToken(tracksToken_, tracksH);
    edm::Handle<reco::MuonCollection> muonsH;
    evt.getByToken(muonsToken_, muonsH);

    const auto& muons = *muonsH;
    const size_t n = muons.size();

    std::vector<int32_t> trackIdx(n, -1);
    std::vector<uint8_t> isMuon(n, 0);
    std::vector<uint8_t> isTrackerMuon(n, 0);
    std::vector<int32_t> muonDtHits(n, 0);
    std::vector<int32_t> muonCscHits(n, 0);
    std::vector<int32_t> muonType(n, 0);

    for (size_t im = 0; im < n; ++im) {
      const auto& mu = muons[im];

      // Direct index lookup via Ref::key(). Only valid if the
      // muon's inner track was built from the SAME collection as tracks (standard
      // convention: reco::Muon::track() -> generalTracks). Falls back to -1 (unresolved)
      // otherwise, same sentinel convention as the rest of this table's index columns.
      reco::TrackRef tref = mu.track();
      if (tref.isNonnull() && tref.id() == tracksH.id()) {
        trackIdx[im] = static_cast<int32_t>(tref.key());
      }

      reco::MuonRef mref(muonsH, im);
      isMuon[im] = PFMuonAlgo::isMuon(mref) ? 1 : 0;
      isTrackerMuon[im] = mu.isTrackerMuon() ? 1 : 0;

      if (mu.standAloneMuon().isNonnull()) {
        const auto& st = mu.standAloneMuon();
        muonDtHits[im] = st->hitPattern().numberOfValidMuonDTHits();
        muonCscHits[im] = st->hitPattern().numberOfValidMuonCSCHits();
      }
      muonType[im] = mu.type();
    }

    auto table = std::make_unique<nanoaod::FlatTable>(n, name_, /*singleton*/ false, /*extension*/ false);
    table->addColumn<int32_t>("trackIdx", trackIdx, "Index of the associated GeneralTrack (-1 if unresolved)");
    table->addColumn<uint8_t>("isMuon", isMuon, "1 if PFMuonAlgo::isMuon, 0 otherwise");
    table->addColumn<uint8_t>("isTrackerMuon", isTrackerMuon, "1 if tracker muon, 0 otherwise");
    table->addColumn<int32_t>("muonDtHits", muonDtHits, "Standalone-muon DT hits (0 if no standalone muon)");
    table->addColumn<int32_t>("muonCscHits", muonCscHits, "Standalone-muon CSC hits (0 if no standalone muon)");
    table->addColumn<int32_t>("muonType", muonType, "reco::Muon::type() bitmask");
    table->setDoc(doc_);
    evt.put(std::move(table));
  }

private:
  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonsToken_;
  const std::string name_;
  const std::string doc_;
};

DEFINE_FWK_MODULE(TrackMuonTableProducer);
