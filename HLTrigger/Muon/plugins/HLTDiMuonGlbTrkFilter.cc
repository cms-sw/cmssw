#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "HLTDiMuonGlbTrkFilter.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Math/interface/deltaR.h"

HLTDiMuonGlbTrkFilter::HLTDiMuonGlbTrkFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  m_muonsTag = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
  m_muonsToken = consumes<reco::MuonCollection>(m_muonsTag);
  m_candsTag = iConfig.getParameter<edm::InputTag>("inputCandCollection");
  m_candsToken = consumes<reco::RecoChargedCandidateCollection>(m_candsTag);
  m_minTrkHits = iConfig.getParameter<int>("minTrkHits");
  m_minMuonHits = iConfig.getParameter<int>("minMuonHits");
  m_maxNormalizedChi2 = iConfig.getParameter<double>("maxNormalizedChi2");
  m_minDR = iConfig.getParameter<double>("minDR");
  m_allowedTypeMask = iConfig.getParameter<unsigned int>("allowedTypeMask");
  m_requiredTypeMask = iConfig.getParameter<unsigned int>("requiredTypeMask");
  m_trkMuonId = muon::SelectionType(iConfig.getParameter<unsigned int>("trkMuonId"));
  m_minPtMuon1 = iConfig.getParameter<double>("minPtMuon1");
  m_minPtMuon2 = iConfig.getParameter<double>("minPtMuon2");
  m_maxEtaMuon = iConfig.getParameter<double>("maxEtaMuon");
  m_maxYDimuon = iConfig.getParameter<double>("maxYDimuon");
  m_minMass = iConfig.getParameter<double>("minMass");
  m_maxMass = iConfig.getParameter<double>("maxMass");
  m_chargeOpt = iConfig.getParameter<int>("ChargeOpt");
  m_maxDCAMuMu = iConfig.getParameter<double>("maxDCAMuMu");
  m_maxdEtaMuMu = iConfig.getParameter<double>("maxdEtaMuMu");
}

void HLTDiMuonGlbTrkFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputMuonCollection", edm::InputTag(""));
  desc.add<edm::InputTag>("inputCandCollection", edm::InputTag(""));
  desc.add<int>("minTrkHits", -1);
  desc.add<int>("minMuonHits", -1);
  desc.add<double>("maxNormalizedChi2", 1e99);
  desc.add<double>("minDR", 0.1);
  desc.add<unsigned int>("allowedTypeMask", 255);
  desc.add<unsigned int>("requiredTypeMask", 0);
  desc.add<unsigned int>("trkMuonId", 0);
  desc.add<double>("minPtMuon1", 17);
  desc.add<double>("minPtMuon2", 8);
  desc.add<double>("maxEtaMuon", 1e99);
  desc.add<double>("maxYDimuon", 1e99);
  desc.add<double>("minMass", 1);
  desc.add<double>("maxMass", 1e99);
  desc.add<int>("ChargeOpt", 0);
  desc.add<double>("maxDCAMuMu", 1e99);
  desc.add<double>("maxdEtaMuMu", 1e99);
  descriptions.add("hltDiMuonGlbTrkFilter", desc);
}

bool HLTDiMuonGlbTrkFilter::hltFilter(edm::Event& iEvent,
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
  std::vector<unsigned int> filteredMuons;
  for (unsigned int i = 0; i < muons->size(); ++i) {
    const reco::Muon& muon(muons->at(i));
    if ((muon.type() & m_allowedTypeMask) == 0)
      continue;
    if ((muon.type() & m_requiredTypeMask) != m_requiredTypeMask)
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
    if (muon.isTrackerMuon() && !muon::isGoodMuon(muon, m_trkMuonId))
      continue;
    if (muon.pt() < std::min(m_minPtMuon1, m_minPtMuon2))
      continue;
    if (std::abs(muon.eta()) > m_maxEtaMuon)
      continue;
    filteredMuons.push_back(i);
  }

  unsigned int npassed(0);
  std::set<unsigned int> mus;
  if (filteredMuons.size() > 1) {
    // Needed for DCA calculation
    edm::ESHandle<MagneticField> bFieldHandle;
    if (m_maxDCAMuMu < 100.)
      iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
    for (unsigned int i = 0; i < filteredMuons.size() - 1; ++i)
      for (unsigned int j = i + 1; j < filteredMuons.size(); ++j) {
        const reco::Muon& mu1(muons->at(filteredMuons.at(i)));
        const reco::Muon& mu2(muons->at(filteredMuons.at(j)));
        if (std::max(mu1.pt(), mu2.pt()) > std::max(m_minPtMuon1, m_minPtMuon2) &&
            std::abs(mu2.eta() - mu1.eta()) < m_maxdEtaMuMu && deltaR(mu1, mu2) > m_minDR &&
            (mu1.p4() + mu2.p4()).mass() > m_minMass && (mu1.p4() + mu2.p4()).mass() < m_maxMass &&
            std::abs((mu1.p4() + mu2.p4()).Rapidity()) < m_maxYDimuon) {
          if (m_chargeOpt < 0) {
            if (mu1.charge() * mu2.charge() > 0)
              continue;
          } else if (m_chargeOpt > 0) {
            if (mu1.charge() * mu2.charge() < 0)
              continue;
          }
          if (m_maxDCAMuMu < 100.) {
            reco::TrackRef tk1 = mu1.get<reco::TrackRef>();
            reco::TrackRef tk2 = mu2.get<reco::TrackRef>();
            reco::TransientTrack mu1TT(*tk1, &(*bFieldHandle));
            reco::TransientTrack mu2TT(*tk2, &(*bFieldHandle));
            TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
            TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
            if (mu1TS.isValid() && mu2TS.isValid()) {
              ClosestApproachInRPhi cApp;
              cApp.calculate(mu1TS.theState(), mu2TS.theState());
              if (!cApp.status() || cApp.distance() > m_maxDCAMuMu)
                continue;
            }
          }
          mus.insert(filteredMuons.at(i));
          mus.insert(filteredMuons.at(j));
          npassed++;
        }
      }
  }

  for (unsigned int mu : mus)
    filterproduct.addObject(trigger::TriggerMuon, reco::RecoChargedCandidateRef(cands, mu));

  return npassed > 0;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDiMuonGlbTrkFilter);
