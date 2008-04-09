#include "PhysicsTools/PFCandProducer/interface/PFTopProjector.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;

PFTopProjector::PFTopProjector(const edm::ParameterSet& iConfig) {
  
  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  inputTagPileUpPFCandidates_ 
    = iConfig.getParameter<InputTag>("PileUpPFCandidates");

  inputTagIsolatedPFCandidates_ 
    = iConfig.getParameter<InputTag>("IsolatedPFCandidates");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  produces<reco::PFCandidateCollection>();
  

  
  if(verbose_) {    
    ostringstream  msg;
    msg<<"input PFCandidateCollection         : "
       <<inputTagPFCandidates_<<endl
       <<"input PileUpPFCandidateCollection   : "
       <<inputTagPileUpPFCandidates_<<endl 
       <<"input IsolatedPFCandidateCollection : "
       <<inputTagIsolatedPFCandidates_<<endl;     
    LogInfo("PFTopProjector")<<msg.str();
  }
}



PFTopProjector::~PFTopProjector() { }



void PFTopProjector::beginJob(const edm::EventSetup & es) { }


void PFTopProjector::produce(Event& iEvent, 
			  const EventSetup& iSetup) {
  
  LogDebug("PFTopProjector")<<"START event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  fetchCollection(pfCandidates, 
		  inputTagPFCandidates_, 
		  iEvent );

  
  Handle<PileUpPFCandidateCollection> pfPileUpCandidates;
  fetchCollection(pfPileUpCandidates, 
		  inputTagPileUpPFCandidates_, 
		  iEvent );

  
  Handle<IsolatedPFCandidateCollection> pfIsolatedCandidates;
  fetchCollection(pfIsolatedCandidates, 
		  inputTagIsolatedPFCandidates_, 
		  iEvent );

  
  auto_ptr< reco::PFCandidateCollection > 
    pOutput( new reco::PFCandidateCollection ); 
  

  vector<bool> masked( pfCandidates->size(), false);
    
  


  if( pfPileUpCandidates.isValid() ) {
    const PileUpPFCandidateCollection& pileUps = *pfPileUpCandidates;
    
    for(unsigned i=0; i<pileUps.size(); i++) {
      
      PFCandidateRef parent = pileUps[i].parent();
      
      if( parent.id() != pfCandidates.id() )  
	assert(0);

      unsigned indexOfParent = parent.key();
      masked[indexOfParent] = true;
    }
  }
  
  if( pfIsolatedCandidates.isValid() ) {
    const IsolatedPFCandidateCollection& isolated = *pfIsolatedCandidates;
    
    for(unsigned i=0; i<isolated.size(); i++) {
      
      PFCandidateRef parent = isolated[i].parent();
      assert( !parent.isNull() );

      unsigned indexOfParent = parent.key();
      
      if( parent.id() != pfCandidates.id() ) {
	PFCandidateRef parentOfParent = parent->parent();
	assert( !parentOfParent.isNull() );
	
	if( parentOfParent.id() != pfCandidates.id() ) 
	  assert(0);
	else
	  indexOfParent = parentOfParent.key();
	  
      }

      masked[indexOfParent] = true;
    }
  }
  

  assert( pfCandidates.isValid() );
  const PFCandidateCollection& inCands = *pfCandidates;

  for(unsigned i=0; i<inCands.size(); i++) {
    if(masked[i]) continue;
    pOutput->push_back( inCands[i] );
  }
  

  iEvent.put( pOutput );
  
  LogDebug("PFTopProjector")<<"STOP event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
}




