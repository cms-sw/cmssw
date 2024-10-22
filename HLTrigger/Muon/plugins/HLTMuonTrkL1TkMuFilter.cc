#include "HLTMuonTrkL1TkMuFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDMException.h"

HLTMuonTrkL1TkMuFilter::HLTMuonTrkL1TkMuFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  m_muonsTag = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
  m_muonsToken = consumes<reco::MuonCollection>(m_muonsTag);
  m_candsTag = iConfig.getParameter<edm::InputTag>("inputCandCollection");
  m_candsToken = consumes<reco::RecoChargedCandidateCollection>(m_candsTag);
  m_l1GTAlgoBlockTag = iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag");
  m_algoBlockToken = consumes<l1t::P2GTAlgoBlockMap>(m_l1GTAlgoBlockTag);
  m_l1GTAlgoNames = iConfig.getParameter<std::vector<std::string>>("l1GTAlgoNames");
  m_minTrkHits = iConfig.getParameter<int>("minTrkHits");
  m_minMuonHits = iConfig.getParameter<int>("minMuonHits");
  m_minMuonStations = iConfig.getParameter<int>("minMuonStations");
  m_maxNormalizedChi2 = iConfig.getParameter<double>("maxNormalizedChi2");
  m_minPt = iConfig.getParameter<double>("minPt");
  m_minN = iConfig.getParameter<unsigned int>("minN");
  m_maxAbsEta = iConfig.getParameter<double>("maxAbsEta");
}

void HLTMuonTrkL1TkMuFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputMuonCollection", edm::InputTag(""));
  desc.add<edm::InputTag>("inputCandCollection", edm::InputTag(""));
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag(""));
  desc.add<std::vector<std::string>>("l1GTAlgoNames", {});
  desc.add<int>("minTrkHits", -1);
  desc.add<int>("minMuonHits", -1);
  desc.add<int>("minMuonStations", -1);
  desc.add<double>("maxNormalizedChi2", 1e99);
  desc.add<unsigned int>("trkMuonId", 0);
  desc.add<double>("minPt", 24);
  desc.add<unsigned int>("minN", 1);
  desc.add<double>("maxAbsEta", 1e99);
  descriptions.add("hltMuonTrkL1TkMuFilter", desc);
}

bool HLTMuonTrkL1TkMuFilter::hltFilter(edm::Event& iEvent,
                                       const edm::EventSetup& iSetup,
                                       trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(m_muonsToken, muons);
  edm::Handle<reco::RecoChargedCandidateCollection> cands;
  iEvent.getByToken(m_candsToken, cands);
  if (saveTags())
    filterproduct.addCollectionTag(m_candsTag);
  if (cands->size() != muons->size())
    throw edm::Exception(edm::errors::Configuration)
        << "Both input collection must be aligned and represent same physical muon objects";

  std::vector<l1t::P2GTCandidateRef> vl1cands;
  bool check_l1match = true;
  if (m_l1GTAlgoBlockTag == edm::InputTag("") || m_l1GTAlgoNames.empty())
    check_l1match = false;
  if (check_l1match) {
    const l1t::P2GTAlgoBlockMap& algos = iEvent.get(m_algoBlockToken);
    for (auto& algoName : m_l1GTAlgoNames) {
      if (algos.count(algoName) > 0 && algos.at(algoName).decisionBeforeBxMaskAndPrescale()) {
        const l1t::P2GTCandidateVectorRef& objects = algos.at(algoName).trigObjects();
        for (const l1t::P2GTCandidateRef& obj : objects) {
          if (obj->objectType() == l1t::P2GTCandidate::ObjectType::GMTTkMuons) {
            vl1cands.push_back(obj);
            LogDebug("HLTMuonTrkL1TkMuFilter") << "Found P2GTCandidate ObjectType::GMTTkMuons" << std::endl;
          }
        }
      }
    }
  }

  float dR2max = 0.3 * 0.3;
  std::vector<unsigned int> filteredMuons;
  for (unsigned int i = 0; i < muons->size(); ++i) {
    const reco::Muon& muon(muons->at(i));
    // check for dR match to L1 muons
    if (check_l1match) {
      bool matchl1 = false;
      for (auto const& l1cand : vl1cands) {
        if (reco::deltaR2(muon.eta(), muon.phi(), l1cand->eta(), l1cand->phi()) < dR2max) {
          matchl1 = true;
          LogDebug("HLTMuonTrkL1TkMuFilterP2GT") << "Matched to L1 muon" << std::endl;
          break;
        }
      }
      if (!matchl1)
        continue;
    }

    if (muon.numberOfMatchedStations() < m_minMuonStations)
      continue;

    if (!muon.innerTrack().isNull()) {
      if (muon.innerTrack()->numberOfValidHits() < m_minTrkHits)
        continue;
    }

    if (!muon.globalTrack().isNull()) {
      if (muon.globalTrack()->normalizedChi2() > m_maxNormalizedChi2)
        continue;
      if (muon.globalTrack()->hitPattern().numberOfValidMuonHits() < m_minMuonHits)
        continue;
    }

    if (muon.pt() < m_minPt)
      continue;

    if (std::abs(muon.eta()) > m_maxAbsEta)
      continue;

    filteredMuons.push_back(i);
  }

  for (std::vector<unsigned int>::const_iterator itr = filteredMuons.begin(); itr != filteredMuons.end(); ++itr)
    filterproduct.addObject(trigger::TriggerMuon, reco::RecoChargedCandidateRef(cands, *itr));

  return filteredMuons.size() >= m_minN;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMuonTrkL1TkMuFilter);
