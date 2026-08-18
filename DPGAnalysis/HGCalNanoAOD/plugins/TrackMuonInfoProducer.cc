// TrackMuonInfoProducer.cc
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

class TrackMuonInfoProducer : public edm::global::EDProducer<> {
public:
  explicit TrackMuonInfoProducer(const edm::ParameterSet& cfg)
      : tracksToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
        muonsToken_(consumes<reco::MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))) {
    produces<edm::ValueMap<int>>("isMuon");
    produces<edm::ValueMap<int>>("isTrackerMuon");
    produces<edm::ValueMap<int>>("muonDtHits");
    produces<edm::ValueMap<int>>("muonCscHits");
    produces<edm::ValueMap<int>>("muonType");
  }

  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup&) const override {
    edm::Handle<std::vector<reco::Track>> tracksH;
    evt.getByToken(tracksToken_, tracksH);
    edm::Handle<reco::MuonCollection> muonsH;
    evt.getByToken(muonsToken_, muonsH);

    const auto& tracks = *tracksH;
    const auto& muons = *muonsH;

    std::vector<int> isMuonV; isMuonV.reserve(tracks.size());
    std::vector<int> isTrackerMuonV; isTrackerMuonV.reserve(tracks.size());
    std::vector<int> muonDtV; muonDtV.reserve(tracks.size());
    std::vector<int> muonCscV; muonCscV.reserve(tracks.size());
    std::vector<int> muonTypeV; muonTypeV.reserve(tracks.size());

    for (size_t i = 0; i < tracks.size(); ++i) {
      reco::TrackRef tref(tracksH, i);
      int muId = -1;
      for (size_t im = 0; im < muons.size(); ++im) {
        if (muons[im].track().isNonnull() && muons[im].track() == tref) { muId = (int)im; break; }
      }
      if (muId == -1) {
        isMuonV.push_back(-1);
        isTrackerMuonV.push_back(-1);
        muonDtV.push_back(-1);
        muonCscV.push_back(-1);
        muonTypeV.push_back(-1);
      } else {
        reco::MuonRef mref(muonsH, muId);
        bool ismu = PFMuonAlgo::isMuon(mref);
        isMuonV.push_back(ismu ? 1 : 0);
        isTrackerMuonV.push_back((*muonsH)[muId].isTrackerMuon() ? 1 : 0);
        int dtHits = 0, cscHits = 0;
        if (mref->standAloneMuon().isNonnull()) {
          auto st = mref->standAloneMuon();
          dtHits = st->hitPattern().numberOfValidMuonDTHits();
          cscHits = st->hitPattern().numberOfValidMuonCSCHits();
        }
        muonDtV.push_back(dtHits);
        muonCscV.push_back(cscHits);
        muonTypeV.push_back(mref->type());
      }
    }

    auto vm_isMuon = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler f_isMuon(*vm_isMuon);
    f_isMuon.insert(tracksH, isMuonV.begin(), isMuonV.end()); f_isMuon.fill();
    evt.put(std::move(vm_isMuon), "isMuon");

    auto vm_isTracker = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler f_isTracker(*vm_isTracker);
    f_isTracker.insert(tracksH, isTrackerMuonV.begin(), isTrackerMuonV.end()); f_isTracker.fill();
    evt.put(std::move(vm_isTracker), "isTrackerMuon");

    auto vm_dt = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler f_dt(*vm_dt);
    f_dt.insert(tracksH, muonDtV.begin(), muonDtV.end()); f_dt.fill();
    evt.put(std::move(vm_dt), "muonDtHits");

    auto vm_csc = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler f_csc(*vm_csc);
    f_csc.insert(tracksH, muonCscV.begin(), muonCscV.end()); f_csc.fill();
    evt.put(std::move(vm_csc), "muonCscHits");

    auto vm_type = std::make_unique<edm::ValueMap<int>>();
    edm::ValueMap<int>::Filler f_type(*vm_type);
    f_type.insert(tracksH, muonTypeV.begin(), muonTypeV.end()); f_type.fill();
    evt.put(std::move(vm_type), "muonType");
  }

private:
  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonsToken_;
};

DEFINE_FWK_MODULE(TrackMuonInfoProducer);
