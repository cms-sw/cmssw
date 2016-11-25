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
  const double          minPtDiffRel_;
  const double          minMuonTrackRelErr_;
  const double          innerTrackRelErr_;
  const double          minMuonPt_;
  const double          segmentCompatibility_;

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
  , minPtDiffRel_         ( iConfig.getParameter<double>  ("minPtDiffRel") )
  , minMuonTrackRelErr_   ( iConfig.getParameter<double>  ("minMuonTrackRelErr") )
  , innerTrackRelErr_     ( iConfig.getParameter<double>  ("innerTrackRelErr") )
  , minMuonPt_            ( iConfig.getParameter<double>  ("minMuonPt") )
  , segmentCompatibility_ ( iConfig.getParameter<double>  ("segmentCompatibility") )
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

    reco::TrackRef bestMuonTrack = muon.muonBestTrack();

    if (debug_) cout<<"BadChargedCandidate test:Muon "<< i << endl;
    // reco::TrackRef innerMuonTrack = muon.innerTrack();
    //  if ( muon.pt() < minMuonPt_ && innerMuonTrack->pt() < minMuonPt_) {
    //  if (debug_) cout <<"skipping the muon because low muon pt" << endl;
    //  continue ; } {
    {
       reco::TrackRef innerMuonTrack = muon.innerTrack();
        if (debug_) cout<<"muon "<<muon.pt()<<endl;
	
        if ( innerMuonTrack.isNull() ) { 
            if (debug_) cout<<"Skipping this muon because it has no inner track"<<endl; 
            continue; 
            };

	if ( muon.pt() < minMuonPt_ && innerMuonTrack->pt() < minMuonPt_) {
          if (debug_) cout <<"skipping the muon because low muon pt" << endl;
          continue;
        }

	// Consider only Global Muons  	
	if (muon.isGlobalMuon() == 0) {
	  if(debug_) cout << "Skipping this muon because not a Global Muon" << endl;
	  continue;
	}

	if (debug_) cout << "SegmentCompatibility :"<< muon::segmentCompatibility(muon) << "RelPtErr:" << bestMuonTrack->ptError()/bestMuonTrack->pt() << endl;
	if (muon::segmentCompatibility(muon) > segmentCompatibility_ && bestMuonTrack->ptError()/bestMuonTrack->pt() < minMuonTrackRelErr_ && innerMuonTrack->ptError()/innerMuonTrack->pt() < innerTrackRelErr_ ) {
	  if (debug_) cout <<"Skipping this muon because segment compatiblity > 0.3 and relErr(best track) <2 and relErr(inner track) < 1 " << endl;
	  continue;
	}

        for ( unsigned j=0; j < pfCandidates->size(); ++j ) {
            const reco::Candidate & pfCandidate = (*pfCandidates)[j];
            // look for charged hadrons
            if (not ( abs(pfCandidate.pdgId()) == 211) ) continue;
            float dr = deltaR( innerMuonTrack->eta(), innerMuonTrack->phi(), pfCandidate.eta(), pfCandidate.phi() );
            float dpt = ( pfCandidate.pt() - innerMuonTrack->pt())/(0.5*(innerMuonTrack->pt() + pfCandidate.pt()));
            if ( (debug_)  and (dr<0.5) ) cout<<" pt(it) "<<innerMuonTrack->pt()<<" candidate "<<pfCandidate.pt()<<" dr "<< dr
                <<" dpt "<<dpt<<endl;
            // require similar pt and small dR , updated to tight comditions and PF check
            if ( ( dr < maxDR_ )  and ( fabs(dpt) < minPtDiffRel_ ) and (muon.isPFMuon()==0) ) {
                    foundBadChargedCandidate = true;
                    cout <<"found bad track!"<<endl; 
                    break;
                }
        }
        if (foundBadChargedCandidate) { break; };
    }
  } // end loop over muonss

  bool pass = !foundBadChargedCandidate;

  if (debug_) cout<<"BadChargedCandidateFilter pass: "<<pass<<endl;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BadChargedCandidateFilter);
