#include "PhysicsTools/PFCandProducer/interface/PFPileUp.h"
#include "PhysicsTools/PFCandProducer/interface/FetchCollection.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;

PFPileUp::PFPileUp(const edm::ParameterSet& iConfig) {
  


  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  produces<reco::PileUpPFCandidateCollection>();
  

//   LogDebug("PFPileUp")
//     <<" input collection : "<<inputTagPFCandidates_ ;
   
}



PFPileUp::~PFPileUp() { }



void PFPileUp::beginJob(const edm::EventSetup & es) { }


void PFPileUp::produce(Event& iEvent, 
			  const EventSetup& iSetup) {
  
//   LogDebug("PFPileUp")<<"START event: "<<iEvent.id().event()
// 			 <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  pfpat::fetchCollection(pfCandidates, 
			 inputTagPFCandidates_, 
			 iEvent );

  // get PFCandidates for isolation
  
  
  auto_ptr< reco::PileUpPFCandidateCollection > 
    pOutput( new reco::PileUpPFCandidateCollection ); 
  
  for( unsigned i=0; i<pfCandidates->size(); i++ ) {
    
    const reco::PFCandidate& cand = (*pfCandidates)[i];
    PFCandidateRef candref(pfCandidates, i);
    
    bool isPileUp = false;
    // just to debug ! all particles with neg charge 
    // are considered to be pile-up
    if( cand.charge()!=0 )
      isPileUp = true; 

    if( isPileUp ) {
      pOutput->push_back( PileUpPFCandidate( candref ) );
    }
    
  }
  
  iEvent.put( pOutput );
  
//   LogDebug("PFPileUp")<<"STOP event: "<<iEvent.id().event()
// 			 <<" in run "<<iEvent.id().run()<<endl;
}

