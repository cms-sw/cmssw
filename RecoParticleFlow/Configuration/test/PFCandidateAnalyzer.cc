#include "RecoParticleFlow/PFProducer/test/PFCandidateAnalyzer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;

PFCandidateAnalyzer::PFCandidateAnalyzer(const edm::ParameterSet& iConfig) {
  


  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  

  LogDebug("PFCandidateAnalyzer")
    <<" input collection : "<<inputTagPFCandidates_ ;
   
}



PFCandidateAnalyzer::~PFCandidateAnalyzer() { }



void PFCandidateAnalyzer::beginJob(const edm::EventSetup & es) { }


void PFCandidateAnalyzer::analyze(const Event& iEvent, 
				  const EventSetup& iSetup) {
  
  LogDebug("PFCandidateAnalyzer")<<"START event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  fetchCandidateCollection(pfCandidates, 
			   inputTagPFCandidates_, 
			   iEvent );

  // get PFCandidates for isolation
  
  
  
  for( unsigned i=0; i<pfCandidates->size(); i++ ) {
    
    const reco::PFCandidate& cand = (*pfCandidates)[i];
    
    if( verbose_ )
      cout<<cand<<endl;
    
  }
    
  LogDebug("PFCandidateAnalyzer")<<"STOP event: "<<iEvent.id().event()
			 <<" in run "<<iEvent.id().run()<<endl;
}


void 
PFCandidateAnalyzer::fetchCandidateCollection(Handle<reco::PFCandidateCollection>& c, 
				      const InputTag& tag, 
				      const Event& iEvent) const {
  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found ) {
    ostringstream  err;
    err<<" cannot get PFCandidates: "
       <<tag<<endl;
    LogError("PFCandidates")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }
  
}



DEFINE_FWK_MODULE(PFCandidateAnalyzer);
