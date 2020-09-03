// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class declaration
//

class BadParticleFilter : public edm::global::EDFilter<> {
public:
  explicit BadParticleFilter(const edm::ParameterSet&);
  ~BadParticleFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::Candidate>> tokenPFCandidates_;
  edm::EDGetTokenT<edm::View<reco::Muon>> tokenMuons_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vtx_;

  const bool taggingMode_;
  int algo_;
  const double maxDR_;
  const double minPtDiffRel_;
  const double minMuonTrackRelErr_;
  const double innerTrackRelErr_;
  const double minDzBestTrack_;
  const double minMuPt_;
  const double segmentCompatibility_;

  double maxDR2_;

  int filterType_;
  enum { kBadPFMuon = 0, kBadPFMuonSummer16, kBadChargedCandidate, kBadChargedCandidateSummer16, kBadPFMuonDz };
};

//
// constructors and destructor
//
BadParticleFilter::BadParticleFilter(const edm::ParameterSet& iConfig)
    : tokenPFCandidates_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
      tokenMuons_(consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      vtx_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("vtx"))),
      taggingMode_(iConfig.getParameter<bool>("taggingMode")),
      maxDR_(iConfig.getParameter<double>("maxDR")),
      minPtDiffRel_(iConfig.getParameter<double>("minPtDiffRel")),
      minMuonTrackRelErr_(iConfig.getParameter<double>("minMuonTrackRelErr")),
      innerTrackRelErr_(iConfig.getParameter<double>("innerTrackRelErr")),
      minDzBestTrack_(iConfig.getParameter<double>("minDzBestTrack")),
      minMuPt_(iConfig.getParameter<double>("minMuonPt")),
      segmentCompatibility_(iConfig.getParameter<double>("segmentCompatibility")) {
  maxDR2_ = maxDR_ * maxDR_;

  std::string filterName = iConfig.getParameter<std::string>("filterType");
  if (filterName == "BadPFMuon")
    filterType_ = kBadPFMuon;
  else if (filterName == "BadPFMuonSummer16")
    filterType_ = kBadPFMuonSummer16;
  else if (filterName == "BadChargedCandidate")
    filterType_ = kBadChargedCandidate;
  else if (filterName == "BadChargedCandidateSummer16")
    filterType_ = kBadChargedCandidateSummer16;
  else if (filterName == "BadPFMuonDz")
    filterType_ = kBadPFMuonDz;
  else {
    throw cms::Exception("BadParticleFilter") << " Filter " << filterName << " is not available, please check name \n";
  }

  algo_ = 0;
  if (filterType_ == kBadPFMuon || filterType_ == kBadPFMuonDz) {
    algo_ = iConfig.getParameter<int>("algo");
  }

  produces<bool>();
}

BadParticleFilter::~BadParticleFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool BadParticleFilter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace std;
  using namespace edm;

  typedef View<reco::Candidate> CandidateView;
  Handle<CandidateView> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_, pfCandidates);

  typedef View<reco::Muon> MuonView;
  Handle<MuonView> muons;
  iEvent.getByToken(tokenMuons_, muons);

  auto const& aPV = iEvent.get(vtx_).at(0).position();

  bool foundBadCandidate = false;

  for (unsigned i = 0; i < muons->size(); ++i) {  // loop over all muons

    const reco::Muon& muon = (*muons)[i];

    reco::TrackRef innerMuonTrack = muon.innerTrack();
    reco::TrackRef bestMuonTrack = muon.muonBestTrack();

    if (innerMuonTrack.isNull()) {
      continue;
    }

    if (filterType_ == kBadChargedCandidate || filterType_ == kBadPFMuon || filterType_ == kBadPFMuonDz) {
      if (muon.pt() < minMuPt_ && innerMuonTrack->pt() < minMuPt_)
        continue;
    }
    if (filterType_ == kBadChargedCandidateSummer16) {
      if (muon.pt() < minMuPt_)
        continue;
    }
    if (filterType_ == kBadPFMuonSummer16) {
      if (innerMuonTrack->pt() < minMuPt_)
        continue;
    }

    // Consider only Global Muons
    if (filterType_ == kBadChargedCandidate || filterType_ == kBadPFMuon || filterType_ == kBadPFMuonDz) {
      if (muon.isGlobalMuon() == 0)
        continue;
    }

    if (filterType_ == kBadPFMuon || filterType_ == kBadPFMuonSummer16 || filterType_ == kBadPFMuonDz) {
      if (!(innerMuonTrack->originalAlgo() == algo_ && innerMuonTrack->algo() == algo_))
        continue;
    }

    if (filterType_ == kBadChargedCandidate || filterType_ == kBadPFMuon || filterType_ == kBadPFMuonDz) {
      if (muon::segmentCompatibility(muon) > segmentCompatibility_ &&
          bestMuonTrack->ptError() / bestMuonTrack->pt() < minMuonTrackRelErr_ &&
          innerMuonTrack->ptError() / innerMuonTrack->pt() < innerTrackRelErr_) {
        if (filterType_ == kBadChargedCandidate || filterType_ == kBadPFMuon) {
          continue;
        }

        if (filterType_ == kBadPFMuonDz) {
          if (fabs(bestMuonTrack->dz(aPV)) < minDzBestTrack_) {
            continue;
          }
        }
      }
    }

    if (filterType_ == kBadChargedCandidateSummer16 || filterType_ == kBadPFMuonSummer16) {
      if (innerMuonTrack->quality(reco::TrackBase::highPurity))
        continue;
      if (!(innerMuonTrack->ptError() / innerMuonTrack->pt() > minMuonTrackRelErr_))
        continue;
    }

    for (unsigned j = 0; j < pfCandidates->size(); ++j) {
      const reco::Candidate& pfCandidate = (*pfCandidates)[j];

      float dr2 = 1000;
      if (filterType_ == kBadChargedCandidate || filterType_ == kBadChargedCandidateSummer16) {
        if (!(std::abs(pfCandidate.pdgId()) == 211))
          continue;
        dr2 = deltaR2(innerMuonTrack->eta(), innerMuonTrack->phi(), pfCandidate.eta(), pfCandidate.phi());
        float dpt = (pfCandidate.pt() - innerMuonTrack->pt()) / (0.5 * (innerMuonTrack->pt() + pfCandidate.pt()));
        if ((dr2 < maxDR2_) && (std::abs(dpt) < minPtDiffRel_) &&
            (filterType_ == kBadChargedCandidateSummer16 || muon.isPFMuon() == 0)) {
          foundBadCandidate = true;
          break;
        }
      }

      if (filterType_ == kBadPFMuon || filterType_ == kBadPFMuonSummer16 || filterType_ == kBadPFMuonDz) {
        if (!((std::abs(pfCandidate.pdgId()) == 13) && (pfCandidate.pt() > minMuPt_)))
          continue;
        dr2 = deltaR2(muon.eta(), muon.phi(), pfCandidate.eta(), pfCandidate.phi());
        if (dr2 < maxDR2_) {
          foundBadCandidate = true;
          break;
        }
      }

      if (foundBadCandidate)
        break;
    }
  }  // end loop over muonss

  bool pass = !foundBadCandidate;

  iEvent.put(std::unique_ptr<bool>(new bool(pass)));

  return taggingMode_ || pass;
}

void BadParticleFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // BadPFMuonFilter
  edm::ParameterSetDescription desc;
  desc.add<double>("innerTrackRelErr", 1.0);
  desc.add<double>("minDzBestTrack", -1.0);
  desc.add<edm::InputTag>("PFCandidates", edm::InputTag("particleFlow"));
  desc.add<std::string>("filterType", "BadPFMuon");
  desc.add<double>("segmentCompatibility", 0.3);
  desc.add<double>("minMuonPt", 100);
  desc.add<int>("algo", 14);
  desc.add<bool>("taggingMode", false);
  desc.add<edm::InputTag>("vtx", edm::InputTag("offlinePrimaryVertices"));
  desc.add<double>("minMuonTrackRelErr", 2.0);
  desc.add<double>("maxDR", 0.001);
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<double>("minPtDiffRel", 0.0);
  descriptions.add("BadPFMuonFilter", desc);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BadParticleFilter);
