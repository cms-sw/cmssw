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

class BadPFMuonFilter : public edm::global::EDFilter<> {
public:
  explicit BadPFMuonFilter(const edm::ParameterSet&);
  ~BadPFMuonFilter();

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
  const double          minPtError_;
  const double          innerTrackRelErr_;
  const double          segmentCompatibility_;

};

//
// constructors and destructor
//
BadPFMuonFilter::BadPFMuonFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , tokenMuons_ ( consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag> ("muons")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode") )
  , debug_                ( iConfig.getParameter<bool>    ("debug") )
  , algo_                 ( iConfig.getParameter<int>  ("algo") )
  , minDZ_                ( iConfig.getParameter<double>  ("minDZ") )
  , minMuPt_              ( iConfig.getParameter<double>  ("minMuPt") )
  , minPtError_           ( iConfig.getParameter<double>  ("minPtError") )
  , innerTrackRelErr_     ( iConfig.getParameter<double>  ("innerTrackRelErr") )
  , segmentCompatibility_ ( iConfig.getParameter<double>  ("segmentCompatibility") )
{
  produces<bool>();
}

BadPFMuonFilter::~BadPFMuonFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
BadPFMuonFilter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
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
    reco::TrackRef globalMuonTrack = muon.globalTrack();
    reco::TrackRef bestMuonTrack = muon.muonBestTrack();
    
    if (debug_) cout << "PF filter muon:" << i << endl ;
    if ( innerMuonTrack.isNull() ) { 
      if (debug_) cout<<"Skipping this muon because it has no inner track"<<endl; 
      continue; 
    };

    if (( innerMuonTrack->pt() < minMuPt_) and (muon.pt() < minMuPt_)) {
            if (debug_) cout<<"Skipping this muon trackPt and globalPt is less than threshold"<<endl; 
       continue;
    }

    // Consider only muons from muonSeededStepOutIn algo
    if (debug_) cout<<"Muon inner track original algo: "<<innerMuonTrack->originalAlgo() << endl;
    if (not ( innerMuonTrack->originalAlgo() == algo_  && innerMuonTrack->algo() == algo_ ) ) {
      if (debug_) cout<<"Skipping this muon because is not coming from the muonSeededStepOutIn"<<endl; 
      continue;
    }

    // Consider only Global Muons
    if (muon.isGlobalMuon() == 0) {
      if(debug_) cout << "Skipping this muon because not a Global Muon" << endl;
      continue;
    }
    

    if (debug_) cout << "SegmentCompatibility :"<< muon::segmentCompatibility(muon) << "RelPtErr:" << bestMuonTrack->ptError()/bestMuonTrack->pt() << endl;    
    if (muon::segmentCompatibility(muon) > segmentCompatibility_ && bestMuonTrack->ptError()/bestMuonTrack->pt() < minPtError_ && innerMuonTrack->ptError()/innerMuonTrack->pt() < innerTrackRelErr_) {
      if (debug_) cout <<"Skipping this muon because segment compatiblity > 0.3 and relErr(best track) <2 and relErr(inner track) <1 " << endl;
     continue;
    }
    
    for ( unsigned j=0; j < pfCandidates->size(); ++j ) {
      const reco::Candidate & pfCandidate = (*pfCandidates)[j];
      // look for pf muon
      if (debug_) cout << "pf pdgID:" << abs(pfCandidate.pdgId()) << "pt:" << pfCandidate.pt() << endl;
      if ( not ( ( abs(pfCandidate.pdgId()) == 13) and (pfCandidate.pt() > minMuPt_) ) ) continue;
      // require small  dR
      float dr = deltaR( muon.eta(), muon.phi(), pfCandidate.eta(), pfCandidate.phi() );
      if (dr < 0.001) {
       	foundBadPFMuon=true;
	if (debug_) cout <<"found bad muon! SC:" << muon::segmentCompatibility(muon) <<endl;
	break;
      }
    }

    if (foundBadPFMuon) { break; };

  }

  bool pass = !foundBadPFMuon;

  if (debug_) cout<< "badPFmuon filter"<<"pass: "<<pass<<endl;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BadPFMuonFilter);
