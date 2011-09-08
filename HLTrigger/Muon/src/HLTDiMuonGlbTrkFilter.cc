#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "HLTrigger/Muon/interface/HLTDiMuonGlbTrkFilter.h"
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

HLTDiMuonGlbTrkFilter::HLTDiMuonGlbTrkFilter(const edm::ParameterSet& iConfig){
  m_muons             = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
  m_cands             = iConfig.getParameter<edm::InputTag>("inputCandCollection");
  m_minTrkHits        = iConfig.getParameter<int>("minTrkHits");
  m_minMuonHits       = iConfig.getParameter<int>("minMuonHits");
  m_maxNormalizedChi2 = iConfig.getParameter<double>("maxNormalizedChi2");
  m_minDR             = iConfig.getParameter<double>("minDR");
  m_allowedTypeMask   = iConfig.getParameter<unsigned int>("allowedTypeMask");
  m_requiredTypeMask  = iConfig.getParameter<unsigned int>("requiredTypeMask");
  m_trkMuonId         = muon::SelectionType(iConfig.getParameter<unsigned int>("trkMuonId"));
  m_minPtMuon1        = iConfig.getParameter<double>("minPtMuon1");
  m_minPtMuon2        = iConfig.getParameter<double>("minPtMuon2");
  m_minMass           = iConfig.getParameter<double>("minMass");
  m_saveTags          = iConfig.getParameter<bool>("saveTags");
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>(); // muons that passed di-muon selection
}

void
HLTDiMuonGlbTrkFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  /*
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL3MuonCandidates"));
  //  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag("hltDiMuonL2PreFiltered0"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  desc.add<bool>("FastAccept",false);
  desc.add<double>("MaxEta",2.5);
  desc.add<int>("MinNhits",0);
  desc.add<double>("MaxDr",2.0);
  desc.add<double>("MaxDz",9999.0);
  desc.add<int>("ChargeOpt",0);
  desc.add<double>("MinPtPair",0.0);
  desc.add<double>("MinPtMax",3.0);
  desc.add<double>("MinPtMin",3.0);
  desc.add<double>("MinInvMass",2.8);
  desc.add<double>("MaxInvMass",3.4);
  desc.add<double>("MinAcop",-1.0);
  desc.add<double>("MaxAcop",3.15);
  desc.add<double>("MinPtBalance",-1.0);
  desc.add<double>("MaxPtBalance",999999.0);
  desc.add<double>("NSigmaPt",0.0);
  desc.add<bool>("saveTags",false);
  desc.add<double>("MaxDCAMuMu",99999.9);
  desc.add<double>("MaxRapidityPair",999999.0);
  desc.add<bool>("CutCowboys",false);
  descriptions.add("hltMuonDimuonL3Filter",desc);
  */
}

bool
HLTDiMuonGlbTrkFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs>  filterproduct(new trigger::TriggerFilterObjectWithRefs(path(),module()));

  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(m_muons,muons);
  edm::Handle<reco::RecoChargedCandidateCollection> cands;
  iEvent.getByLabel(m_cands,cands);
  if ( m_saveTags ) filterproduct->addCollectionTag(m_cands);
  if ( cands->size() != muons->size() )
    throw edm::Exception(edm::errors::Configuration) << "Both input collection must be aligned and represent same physical muon objects";
  std::vector<unsigned int> filteredMuons;
  for ( unsigned int i=0; i<muons->size(); ++i ){
    const reco::Muon& muon(muons->at(i));
    if ( (muon.type() & m_allowedTypeMask) == 0 ) continue;
    if ( (muon.type() & m_requiredTypeMask) != m_requiredTypeMask ) continue;
    if ( !muon.innerTrack().isNull() ){
      if (muon.innerTrack()->numberOfValidHits()<m_minTrkHits) continue;
    }
    if ( !muon.globalTrack().isNull() ){
      if (muon.globalTrack()->normalizedChi2()>m_maxNormalizedChi2) continue;
      if (muon.globalTrack()->hitPattern().numberOfValidMuonHits()<m_minMuonHits) continue;
    }
    if ( muon.isTrackerMuon() && !muon::isGoodMuon(muon,m_trkMuonId) ) continue;
    if ( muon.pt() < std::min(m_minPtMuon1,m_minPtMuon2) ) continue;
    filteredMuons.push_back(i);
  }

  unsigned int npassed(0);
  std::set<unsigned int> mus;
  if ( filteredMuons.size()>1 ){
    for ( unsigned int i=0; i < filteredMuons.size()-1; ++i )
      for ( unsigned int j=i+1; j < filteredMuons.size(); ++j ){
	const reco::Muon& mu1(muons->at(filteredMuons.at(i)));
	const reco::Muon& mu2(muons->at(filteredMuons.at(j)));
	if ( std::max( mu1.pt(), mu2.pt()) > std::max(m_minPtMuon1,m_minPtMuon2) &&
	     deltaR(mu1,mu2)>m_minDR && (mu1.p4() + mu2.p4()).mass() > m_minMass )
	  {
	    mus.insert(filteredMuons.at(i));
	    mus.insert(filteredMuons.at(j));
	    npassed++;
	  }
      }
  }

  for ( std::set<unsigned int>::const_iterator itr = mus.begin(); itr != mus.end(); ++itr )
    filterproduct->addObject(trigger::TriggerMuon, reco::RecoChargedCandidateRef(cands,*itr));
  
  iEvent.put(filterproduct);
  return npassed>0;
}
