


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
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//
// class declaration
//

class MuonBadTrackFilter : public edm::EDFilter {
public:
  explicit MuonBadTrackFilter(const edm::ParameterSet&);
  ~MuonBadTrackFilter();

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection>   tokenPFCandidates_;
  const bool taggingMode_;
  const double          ptMin_;
	const double          chi2Min_;
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
MuonBadTrackFilter::MuonBadTrackFilter(const edm::ParameterSet& iConfig)
  : tokenPFCandidates_ ( consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag> ("PFCandidates")  ))
  , taggingMode_          ( iConfig.getParameter<bool>    ("taggingMode")         )
  , ptMin_                ( iConfig.getParameter<double>        ("ptMin")         )
	, chi2Min_              ( iConfig.getParameter<double>      ("chi2Min")         )
  , debug_                ( iConfig.getParameter<bool>          ("debug")         )
{
  produces<bool>();
}

MuonBadTrackFilter::~MuonBadTrackFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MuonBadTrackFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;

  Handle<reco::PFCandidateCollection>     pfCandidates;
  iEvent.getByToken(tokenPFCandidates_,pfCandidates);

  bool foundBadTrack = false;
  // if ( debug_ ) cout << "starting loop over pfCandidates" << endl;

  for ( unsigned i=0; i<pfCandidates->size(); ++i ) {

    const reco::PFCandidate & cand = (*pfCandidates)[i];
	
		if ( fabs(cand.pdgId()) != 13 ) continue;
    // if ( debug_ ) cout << "Found muon" << std::endl;

    if (cand.pt() < ptMin_) continue;

    if (cand.muonRef().isNull()) continue;
    // if ( debug_ ) cout << "Found valid MuonRef" << std::endl;
		
	  const reco::MuonRef       muon  = cand.muonRef();
		
<<<<<<< HEAD
		if (muon->muonBestTrack().isAvailable()) {
			if (muon->muonBestTrack()->hitPattern().numberOfValidMuonHits() == 0) {
				
				if (muon->globalTrack().isAvailable()) {
					if (muon->globalTrack()->normalizedChi2() > chi2Min_) {
						foundBadTrack = true;
						if ( debug_ ) cout << "globalTrack numberOfValidMuonHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonHits() <<
							" numberOfValidMuonCSCHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonCSCHits() <<
							" numberOfValidMuonDTHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonDTHits() <<
							" normalizedChi2: " << muon->globalTrack()->normalizedChi2() <<	endl;
						if ( debug_ ) cout << "muonBestTrack numberOfValidMuonHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonHits() <<
							" numberOfValidMuonCSCHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonCSCHits() <<
							" numberOfValidMuonDTHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonDTHits() <<
							" normalizedChi2: " << muon->muonBestTrack()->normalizedChi2() <<	endl;
					}
				}

			}
		}

		
		// check if at least one track has good quality
		if (muon->innerTrack().isAvailable()) {
      const double P = muon->innerTrack()->p();
      const double DPt = muon->innerTrack()->ptError();
			if (P != 0) {
				if ( debug_ ) cout << "innerTrack DPt/P: " << DPt/P << endl;
				if (DPt/P < 1) {
					if ( debug_ ) cout << "innerTrack good" << endl;
					continue;
				}
			}
		}
		if (muon->pickyTrack().isAvailable()) {
      const double P = muon->pickyTrack()->p();
      const double DPt = muon->pickyTrack()->ptError();
			if (P != 0) {
				if ( debug_ ) cout << "pickyTrack DPt/P: " << DPt/P << endl;
				if (DPt/P < 1) {
					if ( debug_ ) cout << "pickyTrack good" << endl;
					continue;
				}
			}
		}
		if (muon->globalTrack().isAvailable()) {
      const double P = muon->globalTrack()->p();
      const double DPt = muon->globalTrack()->ptError();
			if (P != 0) {
				if ( debug_ ) cout << "globalTrack DPt/P: " << DPt/P << endl;
				if (DPt/P < 1) {
					if ( debug_ ) cout << "globalTrack good" << endl;
					continue;
				}
			}
=======
		if (!muon->muonBestTrack().isAvailable()) continue;
		if (muon->muonBestTrack()->hitPattern().numberOfValidMuonHits() != 0) continue;
		
		if (!muon->globalTrack().isAvailable()) continue;
		if (muon->globalTrack()->normalizedChi2() > chi2Min_) {
			foundBadTrack = true;
			if ( debug_ ) cout << "globalTrack numberOfValidMuonHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonHits() << 
				" numberOfValidMuonCSCHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonCSCHits() << 
				" numberOfValidMuonDTHits: " << muon->globalTrack()->hitPattern().numberOfValidMuonDTHits() <<
				" normalizedChi2: " << muon->globalTrack()->normalizedChi2() <<	endl;
			if ( debug_ ) cout << "muonBestTrack numberOfValidMuonHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonHits() << 
				" numberOfValidMuonCSCHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonCSCHits() << 
				" numberOfValidMuonDTHits: " << muon->muonBestTrack()->hitPattern().numberOfValidMuonDTHits() <<
				" normalizedChi2: " << muon->muonBestTrack()->normalizedChi2() <<	endl;
>>>>>>> 4373e11f02b851e88d7f422f26f9bbcc697d51e9
		}
		if (muon->tpfmsTrack().isAvailable()) {
      const double P = muon->tpfmsTrack()->p();
      const double DPt = muon->tpfmsTrack()->ptError();
			if (P != 0) {
				if ( debug_ ) cout << "tpfmsTrack DPt/P: " << DPt/P << endl;
				if (DPt/P < 1) {
					if ( debug_ ) cout << "tpfmsTrack good" << endl;
					continue;
				}
			}
		}
		if (muon->dytTrack().isAvailable()) {
      const double P = muon->dytTrack()->p();
      const double DPt = muon->dytTrack()->ptError();
			if (P != 0) {
				if ( debug_ ) cout << "dytTrack DPt/P: " << DPt/P << endl;
				if (DPt/P < 1) {
					if ( debug_ ) cout << "dytTrack good" << endl;
					continue;
				}
			}
		}
		if ( debug_ ) cout << "No tracks are good" << endl;
		foundBadTrack = true;
		
    
  } // end loop over PF candidates


  bool pass = !foundBadTrack;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonBadTrackFilter);
