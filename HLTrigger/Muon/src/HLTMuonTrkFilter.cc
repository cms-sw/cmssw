#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "HLTrigger/Muon/interface/HLTMuonTrkFilter.h"
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

HLTMuonTrkFilter::HLTMuonTrkFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  m_muonsTag          = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
  m_muonsToken        = consumes<reco::MuonCollection>(m_muonsTag);
  m_candsTag          = iConfig.getParameter<edm::InputTag>("inputCandCollection");
  m_candsToken        = consumes<reco::RecoChargedCandidateCollection>(m_candsTag);
  m_minTrkHits        = iConfig.getParameter<int>("minTrkHits");
  m_minMuonHits       = iConfig.getParameter<int>("minMuonHits");
  m_minMuonStations   = iConfig.getParameter<int>("minMuonStations");
  m_maxNormalizedChi2 = iConfig.getParameter<double>("maxNormalizedChi2");
  m_allowedTypeMask   = iConfig.getParameter<unsigned int>("allowedTypeMask");
  m_requiredTypeMask  = iConfig.getParameter<unsigned int>("requiredTypeMask");
  m_trkMuonId         = muon::SelectionType(iConfig.getParameter<unsigned int>("trkMuonId"));
  m_minPt             = iConfig.getParameter<double>("minPt");
}

void
HLTMuonTrkFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputMuonCollection",edm::InputTag(""));
  desc.add<edm::InputTag>("inputCandCollection",edm::InputTag(""));
  desc.add<int>("minTrkHits",-1);
  desc.add<int>("minMuonHits",-1);
  desc.add<int>("minMuonStations",-1);
  desc.add<double>("maxNormalizedChi2",1e99);
  desc.add<unsigned int>("allowedTypeMask",255);
  desc.add<unsigned int>("requiredTypeMask",0);
  desc.add<unsigned int>("trkMuonId",0);
  desc.add<double>("minPt",24);
  descriptions.add("hltMuonTrkFilter",desc);
}

bool
HLTMuonTrkFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(m_muonsToken,muons);
  edm::Handle<reco::RecoChargedCandidateCollection> cands;
  iEvent.getByToken(m_candsToken,cands);
  if ( saveTags() ) filterproduct.addCollectionTag(m_candsTag);
  if ( cands->size() != muons->size() )
    throw edm::Exception(edm::errors::Configuration) << "Both input collection must be aligned and represent same physical muon objects";
  std::vector<unsigned int> filteredMuons;
  for ( unsigned int i=0; i<muons->size(); ++i ){
    const reco::Muon& muon(muons->at(i));
    if ( (muon.type() & m_allowedTypeMask) == 0 ) continue;
    if ( (muon.type() & m_requiredTypeMask) != m_requiredTypeMask ) continue;
    if ( muon.numberOfMatchedStations()<m_minMuonStations ) continue;
    if ( !muon.innerTrack().isNull() ){
      if (muon.innerTrack()->numberOfValidHits()<m_minTrkHits) continue;
    }
    if ( !muon.globalTrack().isNull() ){
      if (muon.globalTrack()->normalizedChi2()>m_maxNormalizedChi2) continue;
      if (muon.globalTrack()->hitPattern().numberOfValidMuonHits()<m_minMuonHits) continue;
    }
    if ( muon.isTrackerMuon() && !muon::isGoodMuon(muon,m_trkMuonId) ) continue;
    if ( muon.pt() < m_minPt ) continue;
    filteredMuons.push_back(i);
  }

  for ( std::vector<unsigned int>::const_iterator itr = filteredMuons.begin(); itr != filteredMuons.end(); ++itr )
    filterproduct.addObject(trigger::TriggerMuon, reco::RecoChargedCandidateRef(cands,*itr));

  return filteredMuons.size()>0;
}
