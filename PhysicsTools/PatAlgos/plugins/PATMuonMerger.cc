#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace pat {
  typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}

class PATMuonMerger : public edm::stream::EDProducer<> {
public:
  explicit PATMuonMerger(const edm::ParameterSet& iConfig);
  ~PATMuonMerger() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::InputTag muons_;
  StringCutObjectSelector<pat::Muon, false> muonsCut_;
  edm::InputTag pfCandidate_;
  StringCutObjectSelector<pat::PackedCandidate, false> pfCandidateCut_;
  edm::InputTag lostTrack_;
  StringCutObjectSelector<pat::PackedCandidate, false> lostTrackCut_;

  edm::EDGetTokenT<std::vector<pat::Muon>> muonToken_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate>> pfCandToken_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate>> lostTrackToken_;
};

PATMuonMerger::PATMuonMerger(const edm::ParameterSet& iConfig)
    : muons_(iConfig.getParameter<edm::InputTag>("muons")),
      muonsCut_(iConfig.getParameter<std::string>("muonCut")),
      pfCandidate_(iConfig.getParameter<edm::InputTag>("pfCandidates")),
      pfCandidateCut_(iConfig.getParameter<std::string>("pfCandidatesCut")),
      lostTrack_(iConfig.getParameter<edm::InputTag>("otherTracks")),
      lostTrackCut_(iConfig.getParameter<std::string>("lostTrackCut")) {
  muonToken_ = consumes<std::vector<pat::Muon>>(muons_);
  pfCandToken_ = consumes<std::vector<pat::PackedCandidate>>(pfCandidate_);
  lostTrackToken_ = consumes<std::vector<pat::PackedCandidate>>(lostTrack_);
  produces<std::vector<pat::Muon>>();
}

void PATMuonMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // mergedMuons
  edm::ParameterSetDescription desc;
  desc.add<std::string>("muonCut", "");
  desc.add<edm::InputTag>("otherTracks", edm::InputTag("lostTracks"));
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<std::string>("pfCandidatesCut", "");
  desc.add<edm::InputTag>("muons", edm::InputTag("slimmedMuons"));
  desc.add<std::string>("lostTrackCut", "");
  descriptions.add("mergedMuonsNoCuts", desc);
}

void PATMuonMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<pat::Muon>> muons;
  edm::Handle<std::vector<pat::PackedCandidate>> pfCands;
  edm::Handle<std::vector<pat::PackedCandidate>> lostTracks;

  iEvent.getByToken(muonToken_, muons);
  iEvent.getByToken(pfCandToken_, pfCands);
  iEvent.getByToken(lostTrackToken_, lostTracks);

  auto out = std::make_unique<std::vector<pat::Muon>>();
  out->reserve(muons->size() + pfCands->size() + lostTracks->size());

  // copy all muons
  for (auto& muon : *muons) {
    if (!muonsCut_(muon))
      continue;
    out->push_back(muon);
  }

  // add other pfCandidates, removing duplicates
  for (unsigned int pf = 0; pf < pfCands->size(); ++pf) {
    auto pfCand = pfCands->at(pf);
    if (!pfCandidateCut_(pfCand))
      continue;
    reco::CandidatePtr pfCandPtr(pfCands, pf);
    bool isPFMuon = false;
    for (auto& muon : *muons) {
      for (unsigned int i = 0, n = muon.numberOfSourceCandidatePtrs(); i < n; ++i) {
        reco::CandidatePtr ptr = muon.sourceCandidatePtr(i);
        if (ptr.isNonnull() && ptr == pfCandPtr) {
          isPFMuon = true;
          break;
        }
      }
      if (isPFMuon)
        break;
    }
    if (isPFMuon) {
      continue;
    }

    // now make a reco::Muon and recast to pat::Muon
    double energy = sqrt(pfCand.p() * pfCand.p() + 0.011163691);
    math::XYZTLorentzVector p4(pfCand.px(), pfCand.py(), pfCand.pz(), energy);
    reco::Muon mu(pfCand.charge(), p4, pfCand.vertex());
    pat::Muon aMu(mu);
    out->push_back(aMu);
  }

  // adding now lost tracks, removing duplicates
  for (auto& lostTrack : *lostTracks) {
    if (!lostTrackCut_(lostTrack))
      continue;
    if (std::abs(lostTrack.pdgId()) == 13)
      continue;

    // now make a reco::Muon and recast to pat::Muon
    double energy = sqrt(lostTrack.p() * lostTrack.p() + 0.011163691);
    math::XYZTLorentzVector p4(lostTrack.px(), lostTrack.py(), lostTrack.pz(), energy);
    reco::Muon mu(lostTrack.charge(), p4, lostTrack.vertex());
    pat::Muon aMu(mu);
    out->push_back(aMu);
  }
  iEvent.put(std::move(out));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATMuonMerger);
