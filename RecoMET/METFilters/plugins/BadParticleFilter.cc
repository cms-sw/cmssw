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

//
// class declaration
//

class BadParticleFilter : public edm::global::EDFilter<> {
public:
  explicit BadParticleFilter(const edm::ParameterSet&);
  ~BadParticleFilter();

private:
  virtual bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::Candidate> >   tokenPFCandidates_;
  edm::EDGetTokenT<edm::View<reco::Muon> >   tokenMuons_;

  const bool taggingMode_;
  int             algo_;
  const double          maxDR_;
  const double          minPtDiffRel_;
  const double          minMuonTrackRelErr_;
  const double          innerTrackRelErr_;
  const double          minMuPt_;
  const double          segmentCompatibility_;

  double maxDR2_;
  
  int filterType_;
  enum {kBadPFMuon=0,kBadPFMuonSummer16,kBadChargedCandidate,kBadChargedCandidateSummer16};
};

//
// constructors and destructor
//
BadParticleFilter::BadParticleFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , tokenMuons_ ( consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag> ("muons")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode") )
  , maxDR_                ( iConfig.getParameter<double>  ("maxDR") )
  , minPtDiffRel_         ( iConfig.getParameter<double>  ("minPtDiffRel") )
  , minMuonTrackRelErr_   ( iConfig.getParameter<double>  ("minMuonTrackRelErr") )
  , innerTrackRelErr_     ( iConfig.getParameter<double>  ("innerTrackRelErr") )
  , minMuPt_            ( iConfig.getParameter<double>  ("minMuonPt") )
  , segmentCompatibility_ ( iConfig.getParameter<double>  ("segmentCompatibility") )
{
  maxDR2_=maxDR_*maxDR_;

  std::string filterName=iConfig.getParameter<std::string>("filterType");
  if(filterName=="BadPFMuon") filterType_=kBadPFMuon;
  else if(filterName=="BadPFMuonSummer16") filterType_=kBadPFMuonSummer16;
  else if(filterName=="BadChargedCandidate") filterType_=kBadChargedCandidate;
  else if(filterName=="BadChargedCandidateSummer16") filterType_=kBadChargedCandidateSummer16;
  else {
    throw cms::Exception("BadParticleFilter")<<" Filter "<<filterName<<" is not available, please check name \n"; 
  }

  algo_=0;
  if(filterType_==kBadPFMuon) {
    algo_=iConfig.getParameter<int>("algo");
  }

  produces<bool>();
}

BadParticleFilter::~BadParticleFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
BadParticleFilter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace std;
  using namespace edm;

  typedef View<reco::Candidate> CandidateView;
  Handle<CandidateView> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  typedef View<reco::Muon> MuonView;
  Handle<MuonView> muons;
  iEvent.getByToken(tokenMuons_,muons);

  bool foundBadCandidate = false;

  for ( unsigned i=0; i < muons->size(); ++i ) { // loop over all muons

    const reco::Muon & muon = (*muons)[i];

    reco::TrackRef innerMuonTrack = muon.innerTrack();
    reco::TrackRef bestMuonTrack = muon.muonBestTrack();
 
    if(innerMuonTrack.isNull() ) { 
      continue; 
    }
    
    if(filterType_==kBadChargedCandidate || filterType_==kBadPFMuon) {
      if(muon.pt()<minMuPt_ && innerMuonTrack->pt() < minMuPt_) continue;
    }
    if(filterType_==kBadChargedCandidateSummer16) {
      if(muon.pt()<minMuPt_) continue;
    }
    if(filterType_==kBadPFMuonSummer16) {
      if(innerMuonTrack->pt() < minMuPt_) continue;
    }

    // Consider only Global Muons  	
    if(filterType_==kBadChargedCandidate || filterType_==kBadPFMuon) {
      if(muon.isGlobalMuon() == 0) continue;
    }

    
    if(filterType_==kBadPFMuon || filterType_==kBadPFMuonSummer16) {
      if(! (innerMuonTrack->originalAlgo() == algo_  &&
	    innerMuonTrack->algo() == algo_ ) ) continue;
    }

    if(filterType_==kBadChargedCandidate || filterType_==kBadPFMuon) {
      if(muon::segmentCompatibility(muon) > segmentCompatibility_ &&
	 bestMuonTrack->ptError()/bestMuonTrack->pt() < minMuonTrackRelErr_ &&
	 innerMuonTrack->ptError()/innerMuonTrack->pt() < innerTrackRelErr_ ) {
	continue;
      }
    }
    if(filterType_==kBadChargedCandidateSummer16 || filterType_==kBadPFMuonSummer16) {
      if(innerMuonTrack->quality(reco::TrackBase::highPurity)) continue;
      if(!(innerMuonTrack->ptError()/innerMuonTrack->pt() > minMuonTrackRelErr_) ) continue;
    }

    
    for(unsigned j=0;j<pfCandidates->size();++j ) {
      const reco::Candidate & pfCandidate = (*pfCandidates)[j];

      float dr2=1000;
      if(filterType_==kBadChargedCandidate || filterType_==kBadChargedCandidateSummer16) {
	if(!(std::abs(pfCandidate.pdgId()) == 211) ) continue;
	dr2 = deltaR2( innerMuonTrack->eta(), innerMuonTrack->phi(), pfCandidate.eta(), pfCandidate.phi() );
	float dpt = ( pfCandidate.pt() - innerMuonTrack->pt())/(0.5*(innerMuonTrack->pt() + pfCandidate.pt()));
	if( (dr2<maxDR2_) && (std::abs(dpt)<minPtDiffRel_) &&
	    (filterType_==kBadChargedCandidateSummer16 || muon.isPFMuon()==0) ) {
	  foundBadCandidate = true;
	  break;
	}
	
      }
      
      if(filterType_==kBadPFMuon || filterType_==kBadPFMuonSummer16) {
	if(!((std::abs(pfCandidate.pdgId()) == 13) && (pfCandidate.pt() > minMuPt_) ) ) continue;
	dr2 = deltaR2( muon.eta(), muon.phi(), pfCandidate.eta(), pfCandidate.phi() );
	if (dr2 < maxDR2_) {
	  foundBadCandidate=true;
	  break;
	}
      }

      if(foundBadCandidate) break;

    }
  } // end loop over muonss

  bool pass = !foundBadCandidate;

  iEvent.put( std::unique_ptr<bool>(new bool(pass)) );
    
  return taggingMode_ || pass;


}




//define this as a plug-in
DEFINE_FWK_MODULE(BadParticleFilter);
