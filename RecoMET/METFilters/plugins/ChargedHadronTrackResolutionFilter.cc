


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
//
// class declaration
//

class ChargedHadronTrackResolutionFilter : public edm::global::EDFilter<> {
public:
  explicit ChargedHadronTrackResolutionFilter(const edm::ParameterSet&);
  ~ChargedHadronTrackResolutionFilter();

private:
  virtual bool filter(edm::StreamID iID, edm::Event&, const edm::EventSetup&) const override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection>   tokenPFCandidates_;
  edm::EDGetTokenT<reco::PFMETCollection>   tokenPFMET_;
  const bool taggingMode_;
  const double          ptMin_;
  const double          metSignifMin_;
  const double          p1_;
  const double          p2_;
  const double          p3_;
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
  , tokenPFMET_ (        consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag> ("PFMET")  )) 
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode")         )
  , ptMin_                ( iConfig.getParameter<double>        ("ptMin")         )
  , metSignifMin_         ( iConfig.getParameter<double>        ("MetSignifMin")  )
  , p1_                   ( iConfig.getParameter<double>        ("p1")            )
  , p2_                   ( iConfig.getParameter<double>        ("p2")            )
  , p3_                   ( iConfig.getParameter<double>        ("p3")            )
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
ChargedHadronTrackResolutionFilter::filter(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection>     pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);
  Handle<reco::PFMETCollection>     pfMET;
  iEvent.getByToken(tokenPFMET_,pfMET);

  bool foundBadTrack = false;
  if ( debug_ ) cout << "starting loop over pfCandidates" << endl;

  for ( unsigned i=0; i<pfCandidates->size(); ++i ) {

    const reco::PFCandidate & cand = (*pfCandidates)[i];
  
    if ( std::abs(cand.pdgId()) == 211 ) {

      if (cand.trackRef().isNull()) continue;

      const reco::TrackRef trackref = cand.trackRef();
      const double Pt = trackref->pt();
      const double DPt = trackref->ptError();
      if (Pt < ptMin_) continue;
      if ( debug_ ) cout << "charged hadron track pT > " << Pt << " GeV - " << " dPt: " << DPt << " GeV - algorithm: "  << trackref->algo() << std::endl;

      const double P = trackref->p();
      
      const unsigned int LostHits = trackref->numberOfLostHits();

      if ( (DPt/Pt) > (p1_ * sqrt(p2_*p2_/P+p3_*p3_) / (1.+LostHits)) ) {
        
        const double MET_px = pfMET->begin()->px();
        const double MET_py = pfMET->begin()->py();
        const double MET_et = pfMET->begin()->et();
        const double MET_sumEt = pfMET->begin()->sumEt();
        const double hadron_px = cand.px();
        const double hadron_py = cand.py();
        if (MET_sumEt == 0) continue;
        const double MET_signif = MET_et/MET_sumEt;
        const double MET_et_corr = sqrt( (MET_px + hadron_px)*(MET_px + hadron_px) + (MET_py + hadron_py)*(MET_py + hadron_py) );
        const double MET_signif_corr = MET_et_corr/MET_sumEt;
        if ( debug_ ) std::cout << "MET signif before: " << MET_signif << " - after: " << MET_signif_corr << " - reduction factor: " << MET_signif/MET_signif_corr << endl;

        if ( (MET_signif/MET_signif_corr) > metSignifMin_ ) {

          foundBadTrack = true;

          if ( debug_ ) {
            cout << cand << endl;
            cout << "charged hadron \t" << "track pT = " << Pt << " +/- " << DPt;
            cout << endl;
            cout << "MET: " << MET_et << " MET phi: " << pfMET->begin()->phi()<<
              " MET sumet: " << MET_sumEt << " MET significance: " << MET_et/MET_sumEt << endl;
            cout << "MET_px: " << MET_px << " MET_py: " << MET_py << " hadron_px: " << hadron_px << " hadron_py: " << hadron_py << endl;
            cout << "corrected: " << sqrt( pow((MET_px + hadron_px),2) + pow((MET_py + hadron_py),2)) << endl;
          }
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
