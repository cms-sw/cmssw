


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//
// class declaration
//

class ChargedHadronTrackResolutionFilter : public edm::EDFilter {
public:
  explicit ChargedHadronTrackResolutionFilter(const edm::ParameterSet&);
  ~ChargedHadronTrackResolutionFilter();

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection>   tokenPFCandidates_;
  const bool taggingMode_;
  const double          ptMin_;
  const double          dPtMin_;
  const bool debug_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ChargedHadronTrackResolutionFilter::ChargedHadronTrackResolutionFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode")         )
  , ptMin_                ( iConfig.getParameter<double>        ("ptMin")         )
  , dPtMin_               ( iConfig.getParameter<double>        ("dPtMin")        )
  , debug_                ( iConfig.getParameter<bool>          ("debug")         )
{
  produces<bool>();
}

ChargedHadronTrackResolutionFilter::~ChargedHadronTrackResolutionFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
ChargedHadronTrackResolutionFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection>     pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  bool foundBadTrack = false;
  if ( debug_ ) cout << "starting loop over pfCandidates" << endl;

  for ( unsigned i=0; i<pfCandidates->size(); ++i ) {

    const reco::PFCandidate & cand = (*pfCandidates)[i];
  
    if ( fabs(cand.pdgId()) == 211 ) {
      // if ( debug_ ) cout << "Found charged hadron candidate" << std::endl;

      if (cand.trackRef().isNull()) continue;
      // if ( debug_ ) cout << "Found valid TrackRef" << std::endl;
      const reco::TrackRef trackref = cand.trackRef();
      const double Pt = trackref->pt();
      const double DPt = trackref->ptError();
      if (Pt < ptMin_) continue;
      if ( debug_ ) cout << "charged hadron track pT > " << Pt << " GeV - " << " dPt: " << DPt << " GeV - algorithm: "  << trackref->algo() << std::endl;

      const double P = trackref->p();
      
      const unsigned int LostHits = trackref->numberOfLostHits();

      if ( ((DPt/Pt) > (5 * sqrt(1.20*1.20/P+0.06*0.06) / (1.+LostHits))) && (DPt > dPtMin_) ) {

        foundBadTrack = true;

        if ( debug_ ) {
          cout << cand << endl;
          cout << "charged hadron \t" << "track pT = " << Pt << " +/- " << DPt;
          cout << endl;
        }
      }
    }
  } // end loop over PF candidates

  bool pass = !foundBadTrack;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ChargedHadronTrackResolutionFilter);
