// -*- C++ -*-
// Framework
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

class HLTDiMuonFilter : public edm::EDFilter {
 public:
  HLTDiMuonFilter(const edm::ParameterSet&);
  virtual ~HLTDiMuonFilter(){}
  virtual bool filter(edm::Event&, const edm::EventSetup&);
 private:
  edm::InputTag m_input; 
  int m_minTrkHits;
  int m_minMuonHits;
  unsigned int m_allowedTypeMask;
  unsigned int m_requiredTypeMask;
  double m_maxNormalizedChi2;
  double m_minDR;
  double m_minPtMuon1;
  double m_minPtMuon2;
  int m_minMatches;
  muon::SelectionType m_trkMuonId;
};

HLTDiMuonFilter::HLTDiMuonFilter(const edm::ParameterSet& parameterSet){
  m_input = parameterSet.getParameter<edm::InputTag>("input");
  m_minTrkHits = parameterSet.getParameter<int>("minTrkHits");
  m_minMuonHits = parameterSet.getParameter<int>("minMuonHits");
  m_maxNormalizedChi2 = parameterSet.getParameter<double>("maxNormalizedChi2");
  m_minDR = parameterSet.getParameter<double>("minDR");
  m_allowedTypeMask = parameterSet.getParameter<unsigned int>("allowedTypeMask");
  m_requiredTypeMask = parameterSet.getParameter<unsigned int>("requiredTypeMask");
  m_trkMuonId = muon::SelectionType(parameterSet.getParameter<unsigned int>("trkMuonId"));
  m_minPtMuon1 = parameterSet.getParameter<double>("minPtMuon1");
  m_minPtMuon2 = parameterSet.getParameter<double>("minPtMuon2");
}

bool HLTDiMuonFilter::filter(edm::Event& event, const edm::EventSetup&){
  edm::Handle<reco::MuonCollection> muons;
  event.getByLabel(m_input,muons);
  std::vector<const reco::Muon*> filteredMuons;
  for ( reco::MuonCollection::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon ){
    if ( (muon->type() & m_allowedTypeMask) == 0 ) continue;
    if ( (muon->type() & m_requiredTypeMask) != m_requiredTypeMask ) continue;
    if ( !muon->innerTrack().isNull() ){
      if (muon->innerTrack()->numberOfValidHits()<m_minTrkHits) continue;
    }
    if ( !muon->globalTrack().isNull() ){
      if (muon->globalTrack()->normalizedChi2()>m_maxNormalizedChi2) continue;
      if (muon->globalTrack()->hitPattern().numberOfValidMuonHits()<m_minMuonHits) continue;
    }
    if ( muon->isTrackerMuon() && !muon::isGoodMuon(*muon,m_trkMuonId) ) continue;
    if ( muon->pt() < std::min(m_minPtMuon1,m_minPtMuon2) ) continue;
    filteredMuons.push_back(&*muon);
  }
  for ( std::vector<const reco::Muon*>::const_iterator mu1 = filteredMuons.begin(); mu1!=filteredMuons.end(); ++mu1 )
    for ( std::vector<const reco::Muon*>::const_iterator mu2 = mu1; ++mu2!=filteredMuons.end(); )
      if ( std::max( (*mu1)->pt(), (*mu2)->pt()) > std::max(m_minPtMuon1,m_minPtMuon2) &&
	   deltaR(**mu1,**mu2)>m_minDR ) return true;
  return false;
}
DEFINE_FWK_MODULE(HLTDiMuonFilter);
