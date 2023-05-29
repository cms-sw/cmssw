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
  m_previousCandTag = iConfig.getParameter<edm::InputTag>("previousCandTag");
  m_previousCandToken = consumes<trigger::TriggerFilterObjectWithRefs>(m_previousCandTag);
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
  desc.add<edm::InputTag>("previousCandTag", edm::InputTag(""));
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

  edm::Handle<trigger::TriggerFilterObjectWithRefs> previousLevelCands;
  std::vector<l1t::TrackerMuonRef> vl1cands;

  bool check_l1match = true;
  if (m_previousCandTag == edm::InputTag(""))
    check_l1match = false;
  if (check_l1match) {
    iEvent.getByToken(m_previousCandToken, previousLevelCands);
    previousLevelCands->getObjects(trigger::TriggerL1TkMu, vl1cands);
  }

  std::vector<unsigned int> filteredMuons;
  for (unsigned int i = 0; i < muons->size(); ++i) {
    const reco::Muon& muon(muons->at(i));
    // check for dR match to L1 muons
    if (check_l1match) {
      bool matchl1 = false;
      for (auto const& l1cand : vl1cands) {
        if (deltaR2(muon, *l1cand) < 0.3 * 0.3) {
          matchl1 = true;
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
