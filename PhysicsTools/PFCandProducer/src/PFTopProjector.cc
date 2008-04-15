#include "PhysicsTools/PFCandProducer/interface/PFTopProjector.h"

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

  Handle<PFJetCollection> pfJets;
  fetchCollection(pfJets, 
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
      const PFJet& jet = jets[i];
      const RefToBaseVector<Candidate>& 
	constituents = jet.getJetConstituents();

      if(verbose_) {
	cout<<jet.print()<<endl;
      }

      for(unsigned jCand=0; jCand<constituents.size(); jCand++) {
	CandidateBaseRef baseRef = constituents[jCand];
	CandidateBaseRef ancestor = refToAncestorPFCandidate( baseRef,
							      pfCandidates );
	ProductID id = ancestor.id();
	unsigned index = ancestor.key();

	if(verbose_) 
	  cout<<"jet pf cand "<<i<<", ancestor "<<id<<"/"<<index<<endl;
	masked[index] = true;
      }
    }
  }

  if(verbose_) 
    cout<<"\tPFPU ------ "<<endl;
  

  if( pfPileUpCandidates.isValid() ) {
    
    const PileUpPFCandidateCollection& pileUps = *pfPileUpCandidates;
    
    for(unsigned i=0; i<pileUps.size(); i++) {
      
      PileUpPFCandidateRef isoRef( pfPileUpCandidates, i ); 
      CandidateBaseRef baseRef( isoRef );

      CandidateBaseRef ancestor = refToAncestorPFCandidate( baseRef,
							    pfCandidates );

      ProductID id = ancestor.id();
      unsigned index = ancestor.key();

      if(verbose_) 
	cout<<"pileup pf cand "<<i<<", ancestor "<<id<<"/"<<index<<endl;
      masked[index] = true;
    }
  }
  
  if(verbose_) 
    cout<<"\tPFIso ------ "<<endl;

  if( pfIsolatedCandidates.isValid() ) {

    const IsolatedPFCandidateCollection& isolated = *pfIsolatedCandidates;
    
    for(unsigned i=0; i<isolated.size(); i++) {

      
      IsolatedPFCandidateRef isoRef( pfIsolatedCandidates, i ); 
      CandidateBaseRef baseRef( isoRef );

      CandidateBaseRef ancestor = refToAncestorPFCandidate( baseRef,
							    pfCandidates );   
      ProductID id = ancestor.id();
      unsigned index = ancestor.key();

      if(verbose_) 
	cout<<"isolated pf cand "<<i<<", ancestor "<<id<<"/"<<index<<endl;
      masked[index] = true;
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


CandidateBaseRef   
PFTopProjector::refToAncestorPFCandidate( CandidateBaseRef candRef,
 					 const Handle<PFCandidateCollection> ancestorPFCandidates ) const {

  
//   cout<<"going down from "<<candRef.id()
//       <<"/"<<candRef.key()
//       <<" ancestor id "<<ancestorPFCandidates.id()<<endl;

  CandidateBaseRef parent = candRef->motherRef(); 
//   cout<<"  parent id "<<parent.id()<<endl;
  if(  parent.id() != ancestorPFCandidates.id() ) {
    // not yet at lowest level

    parent = refToAncestorPFCandidate( parent, ancestorPFCandidates);
  }
  
  return parent;
}

