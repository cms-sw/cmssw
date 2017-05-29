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

class BadPFMuonSummer16Filter : public edm::global::EDFilter<> {
public:
  explicit BadPFMuonSummer16Filter(const edm::ParameterSet&);
  ~BadPFMuonSummer16Filter();

private:
  virtual bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::Candidate> >   tokenPFCandidates_;
  edm::EDGetTokenT<edm::View<reco::Muon> >   tokenMuons_;

  const bool taggingMode_;
  const bool debug_;
  const int             algo_;
  const double          minDZ_;
  const double          minMuPt_;
  const double          minTrkPtError_;

};

//
// constructors and destructor
//
BadPFMuonSummer16Filter::BadPFMuonSummer16Filter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , tokenMuons_ ( consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag> ("muons")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode") )
  , debug_                ( iConfig.getParameter<bool>    ("debug") )
  , algo_                 ( iConfig.getParameter<int>  ("algo") )
  , minDZ_                ( iConfig.getParameter<double>  ("minDZ") )
  , minMuPt_              ( iConfig.getParameter<double>  ("minMuPt") )
  , minTrkPtError_        ( iConfig.getParameter<double>  ("minTrkPtError") )
{
  produces<bool>();
}

BadPFMuonSummer16Filter::~BadPFMuonSummer16Filter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
BadPFMuonSummer16Filter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace std;
  using namespace edm;

  typedef View<reco::Candidate> CandidateView;
  Handle<CandidateView> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  typedef View<reco::Muon> MuonView;
  Handle<MuonView> muons;
  iEvent.getByToken(tokenMuons_,muons);

  bool foundBadPFMuon = false;

  for ( unsigned i=0; i < muons->size(); ++i ) { // loop over all muons
    
    const reco::Muon & muon = (*muons)[i];

    reco::TrackRef innerMuonTrack = muon.innerTrack();

    if ( innerMuonTrack.isNull() ) { 
      if (debug_) cout<<"Skipping this muon because it has no inner track"<<endl; 
      continue; 
    };

    if ( innerMuonTrack->pt() < minMuPt_) {
      if (debug_) cout<<"Skipping this muon because inner track pt."<<endl; 
      continue;
    }

    if ( innerMuonTrack->quality(reco::TrackBase::highPurity) ) { 
      if (debug_) cout<<"Skipping this muon because inner track is high purity."<<endl; 
      continue;
    }

    // Consider only muons with large relative pt error
    if (debug_) cout<<"Muon inner track pt rel err: "<<innerMuonTrack->ptError()/innerMuonTrack->pt()<<endl;
    if (not ( innerMuonTrack->ptError()/innerMuonTrack->pt() > minTrkPtError_ ) ) {
      if (debug_) cout<<"Skipping this muon because seems well measured."<<endl; 
      continue;
    }

    // Consider only muons from muonSeededStepOutIn algo
    if (debug_) cout<<"Muon inner track original algo: "<<innerMuonTrack->originalAlgo() << endl;
    if (not ( innerMuonTrack->originalAlgo() == algo_  && innerMuonTrack->algo() == algo_ ) ) {
      if (debug_) cout<<"Skipping this muon because is not coming from the muonSeededStepOutIn"<<endl; 
      continue;
    }
    
    for ( unsigned j=0; j < pfCandidates->size(); ++j ) {
      const reco::Candidate & pfCandidate = (*pfCandidates)[j];
      // look for pf muon
      if ( not ( ( abs(pfCandidate.pdgId()) == 13) and (pfCandidate.pt() > minMuPt_) ) ) continue;
      // require small dR
      float dr = deltaR( muon.eta(), muon.phi(), pfCandidate.eta(), pfCandidate.phi() );
      if( dr < 0.001 ) {
	foundBadPFMuon=true;
	if (debug_) cout <<"found bad muon!"<<endl;
	break;
      }
    }

    if (foundBadPFMuon) { break; };

  }

  bool pass = !foundBadPFMuon;

  if (debug_) cout<<"pass: "<<pass<<endl;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BadPFMuonSummer16Filter);
