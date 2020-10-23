#include <memory>

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace pat {

  class PackedCandidateMuonSelectorProducer : public edm::stream::EDProducer<> {
  public:
    explicit PackedCandidateMuonSelectorProducer(const edm::ParameterSet& iConfig)
        : muonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
          candidateToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("candidates"))),
          candidate2PFToken_(consumes<edm::Association<reco::PFCandidateCollection>>(
              iConfig.getParameter<edm::InputTag>("candidates"))),
          track2LostTrackToken_(consumes<edm::Association<pat::PackedCandidateCollection>>(
              iConfig.getParameter<edm::InputTag>("lostTracks"))),
          muonSelectors_(iConfig.getParameter<std::vector<std::string>>("muonSelectors")),
          muonIDs_(iConfig.getParameter<std::vector<std::string>>("muonIDs")) {
      for (const auto& sel : muonSelectors_) {
        produces<pat::PackedCandidateRefVector>("lostTracks" + sel);
        produces<pat::PackedCandidateRefVector>("pfCandidates" + sel);
      }
      for (const auto& sel : muonIDs_) {
        muonIDMap_[sel] = std::make_unique<StringCutObjectSelector<reco::Muon>>("passed('" + sel + "')");
        produces<pat::PackedCandidateRefVector>("lostTracks" + sel);
        produces<pat::PackedCandidateRefVector>("pfCandidates" + sel);
      }
    }
    ~PackedCandidateMuonSelectorProducer() override = default;

    void produce(edm::Event&, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
    const edm::EDGetTokenT<pat::PackedCandidateCollection> candidateToken_;
    const edm::EDGetTokenT<pat::PackedCandidateCollection> lostTrackToken_;
    const edm::EDGetTokenT<edm::Association<reco::PFCandidateCollection>> candidate2PFToken_;
    const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> track2LostTrackToken_;
    const std::vector<std::string> muonSelectors_;
    const std::vector<std::string> muonIDs_;
    std::map<std::string, std::unique_ptr<StringCutObjectSelector<reco::Muon>>> muonIDMap_;
  };

}  // namespace pat

void pat::PackedCandidateMuonSelectorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& muons = iEvent.get(muonToken_);
  const auto& candidates = iEvent.getHandle(candidateToken_);
  const auto& candidate2PF = iEvent.get(candidate2PFToken_);
  const auto& track2LostTrack = iEvent.get(track2LostTrackToken_);

  std::map<std::string, std::unique_ptr<pat::PackedCandidateRefVector>> lostTrackMap, candMap;
  for (const auto& sel : muonSelectors_) {
    lostTrackMap.emplace(sel, new pat::PackedCandidateRefVector());
    candMap.emplace(sel, new pat::PackedCandidateRefVector());
  }
  for (const auto& sel : muonIDs_) {
    lostTrackMap.emplace(sel, new pat::PackedCandidateRefVector());
    candMap.emplace(sel, new pat::PackedCandidateRefVector());
  }

  // loop over muons
  for (const auto& muon : muons) {
    const auto& muonTrack = muon.innerTrack();
    // ignore muons without high purity inner track
    if (muonTrack.isNull() || !muonTrack->quality(reco::TrackBase::qualityByName("highPurity")))
      continue;

    // find lost track associated to muon
    const auto& lostTrack = track2LostTrack[muonTrack];
    if (lostTrack.isNonnull()) {
      for (const auto& sel : muonSelectors_) {
        if (muon::isGoodMuon(muon, muon::selectionTypeFromString(sel)))
          lostTrackMap[sel]->push_back(lostTrack);
      }
      for (const auto& sel : muonIDs_) {
        if ((*muonIDMap_.at(sel))(muon))
          lostTrackMap[sel]->push_back(lostTrack);
      }
      continue;
    }

    // find PF candidate associated to muon
    for (size_t i = 0; i < candidates->size(); i++) {
      const auto& cand = pat::PackedCandidateRef(candidates, i);
      const auto& candTrack = candidate2PF[cand]->trackRef();
      // check if candidate and muon are compatible
      if (candTrack.isNonnull() && muonTrack == candTrack) {
        for (const auto& sel : muonSelectors_) {
          if (muon::isGoodMuon(muon, muon::selectionTypeFromString(sel)))
            candMap[sel]->push_back(cand);
        }
        for (const auto& sel : muonIDs_) {
          if ((*muonIDMap_.at(sel))(muon))
            candMap[sel]->push_back(cand);
        }
        break;
      }
    }
  }

  for (auto& s : lostTrackMap) {
    iEvent.put(std::move(s.second), "lostTracks" + s.first);
  }
  for (auto& s : candMap) {
    iEvent.put(std::move(s.second), "pfCandidates" + s.first);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::PackedCandidateMuonSelectorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"))->setComment("muon input collection");
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"))
      ->setComment("packed PF candidate input collection");
  desc.add<edm::InputTag>("lostTracks", edm::InputTag("lostTracks"))->setComment("lost track input collection");
  desc.add<std::vector<std::string>>("muonSelectors", {"AllTrackerMuons", "TMOneStationTight"})
      ->setComment("muon selectors");
  desc.add<std::vector<std::string>>("muonIDs", {})->setComment("muon IDs");
  descriptions.add("packedCandidateMuonID", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PackedCandidateMuonSelectorProducer);
