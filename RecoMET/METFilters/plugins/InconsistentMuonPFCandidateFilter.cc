


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

class InconsistentMuonPFCandidateFilter : public edm::EDFilter {
public:
  explicit InconsistentMuonPFCandidateFilter(const edm::ParameterSet&);
  ~InconsistentMuonPFCandidateFilter();
  
private:
  virtual void beginJob() override ;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;
  
      // ----------member data ---------------------------
  
  const edm::InputTag   inputTagPFCandidates_; 
  const double          ptMin_;
  const double          maxPTDiff_;

  const bool taggingMode_, debug_;
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
InconsistentMuonPFCandidateFilter::InconsistentMuonPFCandidateFilter(const edm::ParameterSet& iConfig)
  : inputTagPFCandidates_ ( iConfig.getParameter<edm::InputTag> ("PFCandidates")  )
  , ptMin_                ( iConfig.getParameter<double>        ("ptMin")         )
  , maxPTDiff_            ( iConfig.getParameter<double>        ("maxPTDiff")     )
  , taggingMode_          ( iConfig.getParameter<bool>          ("taggingMode")   )
  , debug_                ( iConfig.getParameter<bool>          ("debug")         )
{
  produces<bool>();
  produces<reco::PFCandidateCollection>("muons");
}

InconsistentMuonPFCandidateFilter::~InconsistentMuonPFCandidateFilter() { }


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
InconsistentMuonPFCandidateFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;  
  using namespace edm;

  Handle<reco::PFCandidateCollection>     pfCandidates;
  iEvent.getByLabel(inputTagPFCandidates_,pfCandidates);
  
  bool    foundMuon = false;

  auto_ptr< reco::PFCandidateCollection > 
    pOutputCandidateCollection( new reco::PFCandidateCollection ); 

  for ( unsigned i=0; i<pfCandidates->size(); i++ ) {
     
    const reco::PFCandidate & cand = (*pfCandidates)[i];

    if ( cand.particleId() != reco::PFCandidate::mu ) continue; 
    if ( cand.pt() < ptMin_ )                         continue; 

    const reco::MuonRef       muon  = cand.muonRef();
    if (  muon->isTrackerMuon()
       && muon->isGlobalMuon()
       && fabs(muon->innerTrack()->pt()/muon->globalTrack()->pt() - 1) <= maxPTDiff_
       )
      continue;

    foundMuon = true;

    pOutputCandidateCollection->push_back( cand ); 

    if ( debug_ ) {
      cout << cand << endl;
      cout << "\t" << "tracker pT = ";
      if (muon->isTrackerMuon())  cout << muon->innerTrack()->pt();
      else                        cout << "(n/a)";
      cout << endl;
      cout << "\t" << "global fit pT = ";
      if (muon->isGlobalMuon())   cout << muon->globalTrack()->pt();
      else                        cout << "(n/a)";
      cout << endl;
    }    
  } // end loop over PF candidates
   
  iEvent.put( pOutputCandidateCollection, "muons" );

  bool pass = !foundMuon;
  
  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}

// ------------ method called once each job just before starting event loop  ------------
void 
InconsistentMuonPFCandidateFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
InconsistentMuonPFCandidateFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(InconsistentMuonPFCandidateFilter);
