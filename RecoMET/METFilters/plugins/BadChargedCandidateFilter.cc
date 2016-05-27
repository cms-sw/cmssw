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

class BadChargedCandidateFilter : public edm::global::EDFilter<> {
public:
  explicit BadChargedCandidateFilter(const edm::ParameterSet&);
  ~BadChargedCandidateFilter();

private:
  virtual bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::Candidate> >   tokenPFCandidates_;
  edm::EDGetTokenT<edm::View<reco::Muon> >   tokenMuons_;

  const bool taggingMode_;
  const bool debug_;
  const double          maxDR_;
  const double          minPtErrorRel_;
  const double          minMuonPt_;

};

//
// constructors and destructor
//
BadChargedCandidateFilter::BadChargedCandidateFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , tokenMuons_ ( consumes<edm::View<reco::Muon> >(iConfig.getParameter<edm::InputTag> ("muons")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode") )
  , debug_                ( iConfig.getParameter<bool>    ("debug") )
  , maxDR_                ( iConfig.getParameter<double>  ("maxDR") )
  , minPtErrorRel_        ( iConfig.getParameter<double>  ("minPtErrorRel") )
  , minMuonPt_            ( iConfig.getParameter<double>  ("minMuonPt") )
{
  produces<bool>();
}

BadChargedCandidateFilter::~BadChargedCandidateFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
BadChargedCandidateFilter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace std;
  using namespace edm;

  typedef View<reco::Candidate> CandidateView;
  Handle<CandidateView> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  typedef View<reco::Muon> MuonView;
  Handle<MuonView> muons;
  iEvent.getByToken(tokenMuons_,muons);

  bool foundBadChargedCandidate = false;

  for ( unsigned i=0; i < muons->size(); ++i ) { // loop over all muons

    const reco::Muon & muon = (*muons)[i];

    if ( muon.pt() > minMuonPt_) {
        reco::TrackRef innerMuonTrack = muon.innerTrack();
        if (debug_) cout<<"muon "<<muon.pt()<<endl;

        if ( innerMuonTrack.isNull() ) { continue; };

        for ( unsigned j=0; j < pfCandidates->size(); ++j ) {
            const reco::Candidate & pfCandidate = (*pfCandidates)[j];
            if (not abs(pfCandidate.pdgId()) == 211) continue;
            if (debug_) cout<<"candidate "<<pfCandidate.pt()<<" dr "<<deltaR( innerMuonTrack->eta(), innerMuonTrack->phi(), pfCandidate.eta(), pfCandidate.phi() )
                <<" dpt "<<( pfCandidate.pt() - innerMuonTrack->pt())/(0.5*(innerMuonTrack->pt() + pfCandidate.pt()))<<endl;
            if ( ( deltaR( innerMuonTrack->eta(), innerMuonTrack->phi(), pfCandidate.eta(), pfCandidate.phi() ) < maxDR_ ) 
                and ( ( pfCandidate.pt() - innerMuonTrack->pt())/(0.5*(innerMuonTrack->pt() + pfCandidate.pt())) > minPtErrorRel_ ) ) {
                    foundBadChargedCandidate = true;
                    break;
                }
        }
        if (foundBadChargedCandidate) { break; };
    }
  } // end loop over muonss

  bool pass = !foundBadChargedCandidate;

  if (debug_) cout<<"pass: "<<pass<<endl;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BadChargedCandidateFilter);
