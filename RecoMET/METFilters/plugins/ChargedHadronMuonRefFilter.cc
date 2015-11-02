


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

//
// class declaration
//

class ChargedHadronMuonRefFilter : public edm::EDFilter {
public:
  explicit ChargedHadronMuonRefFilter(const edm::ParameterSet&);
  ~ChargedHadronMuonRefFilter();

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection>   tokenPFCandidates_;
  const bool taggingMode_;
  const double          ptMin_;
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
ChargedHadronMuonRefFilter::ChargedHadronMuonRefFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode")         )
  , ptMin_                ( iConfig.getParameter<double>        ("ptMin")         )
  , debug_                ( iConfig.getParameter<bool>          ("debug")         )
{
  produces<bool>();
}

ChargedHadronMuonRefFilter::~ChargedHadronMuonRefFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
ChargedHadronMuonRefFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection>     pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  bool foundBadTrack = false;
  if ( debug_ ) cout << "starting loop over pfCandidates" << endl;

  for ( unsigned i=0; i<pfCandidates->size(); ++i ) {

    const reco::PFCandidate & cand = (*pfCandidates)[i];
    
    // if ( fabs(cand.pdgId()) != 211 ) continue;
    // if ( debug_ ) cout << "Found charged hadron" << std::endl;
    
    if (cand.muonRef().isNull()) continue;
    // if ( debug_ ) cout << "Found valid MuonRef" << std::endl;
        
    const reco::TrackRef trackref = cand.trackRef();
    const double Pt = trackref->pt();
    if (Pt < ptMin_) continue;
    // if ( debug_ ) cout << "track pT > " << ptMin_ << " GeV - algorithm: "  << trackref->algo() << std::endl;
          
      const double P = trackref->p();
      const double DPt = trackref->ptError();
      const unsigned int LostHits = trackref->numberOfLostHits();
            
      if ((DPt/Pt) > (5 * sqrt(1.20*1.20/P+0.06*0.06) / (1.+LostHits))) {
              
        foundBadTrack = true;
        
        if ( debug_ ) {
          cout << cand << endl;
          cout << "\t" << "track pT = " << Pt << " +/- " << DPt;
          cout << endl;
        }
      }
    
  } // end loop over PF candidates

  bool pass = !foundBadTrack;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ChargedHadronMuonRefFilter);
