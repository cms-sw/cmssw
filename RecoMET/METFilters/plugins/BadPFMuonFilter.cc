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

  edm::EDGetTokenT<edm::View<reco::PFCandidate> >   tokenPFCandidates_;

  const bool taggingMode_;
  const bool debug_;
  const double          minDZ_;
  const double          minMuPt_;
  const double          minTrkPtError_;

};

//
// constructors and destructor
//
BadPFMuonFilter::BadPFMuonFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<edm::View<reco::PFCandidate> >(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode") )
  , debug_                ( iConfig.getParameter<bool>    ("debug") )
  , minDZ_                ( iConfig.getParameter<double>  ("minDZ") )
  , minMuPt_              ( iConfig.getParameter<double>  ("minMuPt") )
  , minTrkPtError_        ( iConfig.getParameter<double>  ("minTrkPtError") )
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

  typedef View<reco::PFCandidate> CandidateView;
  Handle<CandidateView> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  bool foundBadPFMuon = false;

  for ( unsigned j=0; j < pfCandidates->size(); ++j ) {
      const reco::PFCandidate & pfCandidate = (*pfCandidates)[j];
      // look for charged hadrons
      if ( not ( ( abs(pfCandidate.pdgId()) == 13) and (pfCandidate.pt() > minMuPt_) ) ) continue;
      reco::TrackRef innerMuonTrack = pfCandidate.muonRef()->innerTrack();
      if (innerMuonTrack.isNull() or innerMuonTrack->quality(reco::TrackBase::highPurity)) continue;
      if ( ( abs(innerMuonTrack->dz()) > minDZ_ ) and ( innerMuonTrack->ptError()>minTrkPtError_ ) ) {
            foundBadPFMuon = true;
            break;
      }
  }

  bool pass = !foundBadPFMuon;

  if (debug_) cout<<"pass: "<<pass<<endl;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BadPFMuonFilter);
