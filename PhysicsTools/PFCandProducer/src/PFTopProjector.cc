#include "PhysicsTools/PFCandProducer/interface/PFTopProjector.h"
#include "PhysicsTools/PFCandProducer/interface/FetchCollection.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

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

  inputTagPFJets_ 
    = iConfig.getParameter<InputTag>("PFJets");

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
       <<inputTagIsolatedPFCandidates_<<endl  
       <<"input PFJetCollection : "
       <<inputTagPFJets_<<endl;     
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
  pfpat::fetchCollection(pfCandidates, 
			 inputTagPFCandidates_, 
			 iEvent );

  Handle<PileUpPFCandidateCollection> pfPileUpCandidates;
  pfpat::fetchCollection(pfPileUpCandidates, 
			 inputTagPileUpPFCandidates_, 
			 iEvent );

  
  Handle<IsolatedPFCandidateCollection> pfIsolatedCandidates;
  pfpat::fetchCollection(pfIsolatedCandidates, 
			 inputTagIsolatedPFCandidates_, 
			 iEvent );

  Handle<PFJetCollection> pfJets;
  pfpat::fetchCollection(pfJets, 
			 inputTagPFJets_, 
			 iEvent );

  if(verbose_) {
    cout<<"Top projector: product Ids --------------------"<<endl;
    cout<<"PF      :  "<<pfCandidates.id()<<endl
	<<"PFPU    :  "<<pfPileUpCandidates.id()<<endl
	<<"PFIso   :  "<<pfIsolatedCandidates.id()<<endl
	<<"PFJets  :  "<<pfJets.id()<<endl;
  }

  auto_ptr< reco::PFCandidateCollection > 
    pOutput( new reco::PFCandidateCollection ); 
  

  vector<bool> masked( pfCandidates->size(), false);
    
  assert( pfCandidates.isValid() );

  if(verbose_) 
    cout<<"\tPFJets ------ "<<endl;
  
  if( pfJets.isValid() ) {
    const PFJetCollection& jets = *pfJets;
    
    for(unsigned i=0; i<jets.size(); i++) {

      PFJetRef jetRef(pfJets, i );
      CandidateBaseRef baseRef( jetRef );
      CandidateBaseRefVector ancestors;
      refToAncestorPFCandidates( baseRef,
				 ancestors,
				 pfCandidates );

      maskAncestors( ancestors, masked );
    }
  }

  

  if( pfPileUpCandidates.isValid() ) {
    
    const PileUpPFCandidateCollection& pileUps = *pfPileUpCandidates;
    
    if(verbose_) 
      cout<<"\tPFPU ------ "<<pileUps.size()<<endl;
 
    for(unsigned i=0; i<pileUps.size(); i++) {
      
      PileUpPFCandidateRef isoRef( pfPileUpCandidates, i ); 
      CandidateBaseRef baseRef( isoRef );
      CandidateBaseRefVector ancestors;
      refToAncestorPFCandidates( baseRef,
				 ancestors,
				 pfCandidates );

      maskAncestors( ancestors, masked );
    }
  }
  

  if( pfIsolatedCandidates.isValid() ) {

    const IsolatedPFCandidateCollection& isolated = *pfIsolatedCandidates;

    if(verbose_) 
      cout<<"\tPFIso ------ "<<isolated.size()<<endl;
    
    for(unsigned i=0; i<isolated.size(); i++) {

      
      IsolatedPFCandidateRef isoRef( pfIsolatedCandidates, i ); 
      CandidateBaseRef baseRef( isoRef );
      CandidateBaseRefVector ancestors;
      refToAncestorPFCandidates( baseRef,
				 ancestors,
				 pfCandidates );

      maskAncestors( ancestors, masked );
    }
  }
  
  const PFCandidateCollection& inCands = *pfCandidates;

  for(unsigned i=0; i<inCands.size(); i++) {
    
    if(masked[i]) {
      if(verbose_)
	cout<<"X "<<i<<" "<<inCands[i]<<endl;
      continue;
    }
    else {
      if(verbose_)
	cout<<"O "<<i<<" "<<inCands[i]<<endl;
      PFCandidateRef motherRef( pfCandidates, i );
      pOutput->push_back( inCands[i] );
      pOutput->back().setMotherRef(motherRef);
    }
  }
  

  iEvent.put( pOutput );
  
  LogDebug("PFTopProjector")<<"STOP event: "<<iEvent.id().event()
 			 <<" in run "<<iEvent.id().run()<<endl;
}


void
PFTopProjector::refToAncestorPFCandidates( CandidateBaseRef candRef,
					   CandidateBaseRefVector& ancestors,
					   const Handle<PFCandidateCollection> allPFCandidates ) const {

  

  CandidateBaseRefVector mothers = candRef->motherRefs(); 
 
//   cout<<"going down from "<<candRef.id()
//       <<"/"<<candRef.key()<<" #mothers "<<mothers.size()
//       <<" ancestor id "<<allPFCandidates.id()<<endl;
  
  for(unsigned i=0; i<mothers.size(); i++) {
//     cout<<"  mother id "<<mothers[i].id()<<endl;
    
    CandidateBaseRef mother = mothers[i];
    
    if(  mother.id() != allPFCandidates.id() ) {
      // the mother is not yet at lowest level
      refToAncestorPFCandidates( mother, ancestors, allPFCandidates);
    }
    else {
      // adding mother to the list of ancestors
      ancestors.push_back( mother ); 
    }
  }
}



void PFTopProjector::maskAncestors( const reco::CandidateBaseRefVector& ancestors,
				    std::vector<bool>& masked ) const {
  
  for(unsigned i=0; i<ancestors.size(); i++) {
    unsigned index = ancestors[i].key();
    assert( index<masked.size() );
    
    if(verbose_) {
      ProductID id = ancestors[i].id();
      cout<<"masking "<<i<<", ancestor "<<id<<"/"<<index<<endl;
    }
    masked[index] = true;
  }
}
