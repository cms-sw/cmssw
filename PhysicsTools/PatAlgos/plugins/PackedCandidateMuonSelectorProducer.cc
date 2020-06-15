#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

namespace pat {

  class PackedCandidateMuonSelectorProducer : public edm::stream::EDProducer<> {

    typedef edm::ValueMap<bool> BoolMap;

    public:

      explicit PackedCandidateMuonSelectorProducer(const edm::ParameterSet & iConfig):
          muonToken_(consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
          candidateToken_(consumes<edm::View<pat::PackedCandidate> >(iConfig.getParameter<edm::InputTag>("candidates"))),
          muonSelectors_(iConfig.getParameter<std::vector<std::string> >("muonSelectors")),
          muonIDs_(iConfig.getParameter<std::vector<std::string> >("muonIDs"))
      {
        for (const auto& sel : muonSelectors_) {
          produces<BoolMap>(sel);
        }
        for (const auto& sel : muonIDs_) {
          muonIDMap_[sel].reset(new StringCutObjectSelector<pat::Muon>("passed('"+sel+"')"));
          produces<BoolMap>(sel);
        }
      }
      ~PackedCandidateMuonSelectorProducer() override {};

      void produce(edm::Event&, const edm::EventSetup&) override;

      static void fillDescriptions(edm::ConfigurationDescriptions&);

    private:

      edm::EDGetTokenT<pat::MuonCollection> muonToken_;
      edm::EDGetTokenT<edm::View<pat::PackedCandidate> > candidateToken_;
      std::vector<std::string> muonSelectors_;
      std::vector<std::string> muonIDs_;
      std::map<std::string, std::unique_ptr<StringCutObjectSelector<pat::Muon> > > muonIDMap_;

  };

}

void pat::PackedCandidateMuonSelectorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<pat::MuonCollection> muons;
  edm::Handle<edm::View<pat::PackedCandidate> > candidates;

  iEvent.getByToken(muonToken_, muons);
  iEvent.getByToken(candidateToken_, candidates);

  const auto& nCand = candidates->size();

  std::map<std::string, std::vector<bool> > muonSelMap;
  for (const auto& sel : muonSelectors_) {
    muonSelMap[sel] = std::vector<bool>(nCand, false);
  }
  for (const auto& sel : muonIDs_) {
    muonSelMap[sel] = std::vector<bool>(nCand, false);
  }

  // loop over muons
  for (const auto& muon : *muons) {
    // ignore muons without high purity inner track
    if (muon.innerTrack().isNull() ||
        !muon.innerTrack()->quality(reco::TrackBase::qualityByName("highPurity"))) continue;

    // find candidate associated to muon
    const auto& muonTrack = *muon.innerTrack();
    for (size_t i = 0; i < nCand; i++) {
      const auto& cand = candidates->refAt(i);
      // ignore neutral candidates or without track
      if (cand->charge()==0 || !cand->hasTrackDetails()) continue;

      // check if candidate and muon are compatible
      const auto& candTrack = cand->pseudoTrack();
      if (muonTrack.charge()==candTrack.charge() &&
          muonTrack.numberOfValidHits()==candTrack.numberOfValidHits() &&
          std::abs(muonTrack.eta()-candTrack.eta())<1E-3 &&
          std::abs(muonTrack.phi()-candTrack.phi())<1E-3 &&
          std::abs((muonTrack.pt()-candTrack.pt())/muonTrack.pt())<1E-2) {
        // fill muon selector map
        for (const auto& sel : muonSelectors_) {
          muonSelMap[sel][i] = muon.muonID(sel);
        }
        for (const auto& sel : muonIDs_) {
          muonSelMap[sel][i] = (*muonIDMap_.at(sel))(muon);
        }
        break;
      }
    }
  }

  // fill the value maps
  for (const auto& s : muonSelMap) {
    std::unique_ptr<BoolMap> valueMap(new BoolMap());
    BoolMap::Filler filler(*valueMap);
    filler.insert(candidates, s.second.begin(), s.second.end());
    filler.fill();
    iEvent.put(std::move(valueMap), s.first);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::PackedCandidateMuonSelectorProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muons", edm::InputTag("patMuons"))->setComment("muon input collection");
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"))->setComment("packed candidate input collection");
  desc.add<std::vector<std::string> >("muonSelectors", {})->setComment("muon selectors");
  desc.add<std::vector<std::string> >("muonIDs", {})->setComment("muon IDs");
  descriptions.add("packedPFCandidateMuonID", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PackedCandidateMuonSelectorProducer);
