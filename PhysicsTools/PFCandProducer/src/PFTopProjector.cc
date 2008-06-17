#include "PhysicsTools/PFCandProducer/interface/PFTopProjector.h"
#include "PhysicsTools/PFCandProducer/interface/FetchCollection.h"

#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
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

  inputTagIsolatedElectrons_ 
    = iConfig.getParameter<InputTag>("IsolatedElectrons");

  inputTagIsolatedMuons_ 
    = iConfig.getParameter<InputTag>("IsolatedMuons");

  inputTagPFJets_ 
    = iConfig.getParameter<InputTag>("PFJets");

  inputTagPFTaus_ 
    = iConfig.getParameter<InputTag>("PFTaus");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);


  produces<reco::PFCandidateCollection>();
  produces<reco::PFJetCollection>();
  

  
  if(verbose_) {    
    ostringstream  msg;
    msg<<"input PFCandidateCollection         : "
       <<inputTagPFCandidates_<<endl
       <<"input PileUpPFCandidateCollection   : "
       <<inputTagPileUpPFCandidates_<<endl 
       <<"input IsolatedPFCandidates of type muon : "
       <<inputTagIsolatedMuons_<<endl  
       <<"input IsolatedPFCandidates of type electron : "
       <<inputTagIsolatedElectrons_<<endl  
       <<"input PFJetCollection : "
       <<inputTagPFJets_<<endl
       <<"input PFTauCollection : "
       <<inputTagPFTaus_<<endl;  
    cout<<msg.str()<<endl;
    //     LogInfo("PFTopProjector")<<msg.str();
  }
}



PFTopProjector::~PFTopProjector() { }



void PFTopProjector::beginJob(const edm::EventSetup & es) { }


void PFTopProjector::produce(Event& iEvent, 
			     const EventSetup& iSetup) {
  
 
  
  
  // get the various collections

  // this collection is the collection of PFCandidates to 
  // be masked by the other ones
  Handle<PFCandidateCollection> pfCandidates;
  pfpat::fetchCollection(pfCandidates, 
			 inputTagPFCandidates_, 
			 iEvent );
  edm::ProductID pfCandidatesID = pfCandidates.id();

  // for each object in the following collections, the 
  // PFTopProjector will find and mask all PFCandidates in pfCandidates
  // involved in the construction of this object
  
  // empty input tags can be specified, and will be recognized 
  // by the fetchCollection function
  
  Handle<PileUpPFCandidateCollection> pfPileUpCandidates;
  pfpat::fetchCollection(pfPileUpCandidates, 
			 inputTagPileUpPFCandidates_, 
			 iEvent );

  Handle<IsolatedPFCandidateCollection> pfIsolatedElectrons;
  pfpat::fetchCollection(pfIsolatedElectrons, 
			 inputTagIsolatedElectrons_, 
			 iEvent );

  Handle<IsolatedPFCandidateCollection> pfIsolatedMuons;
  pfpat::fetchCollection(pfIsolatedElectrons, 
			 inputTagIsolatedElectrons_, 
			 iEvent );

  Handle<PFJetCollection> pfJets;
  pfpat::fetchCollection(pfJets, 
			 inputTagPFJets_, 
			 iEvent );

  Handle<PFTauCollection> pfTaus;
  pfpat::fetchCollection(pfTaus, 
			 inputTagPFTaus_, 
			 iEvent );

  if(verbose_) {
    cout<<"Top projector: event "<<iEvent.id().event()<<endl;
    cout<<"product Ids --------------------"<<endl;
    cout<<"PF       :  "<<pfCandidates.id()<<endl
	<<"PFPU     :  "<<pfPileUpCandidates.id()<<endl
	<<"PFIso e  :  "<<pfIsolatedElectrons.id()<<endl
	<<"PFIso mu :  "<<pfIsolatedMuons.id()<<endl
	<<"PFJets   :  "<<pfJets.id()<<endl
	<<"PFTaus   :  "<<pfTaus.id()<<endl;
  }

  // output PFCandidate collection
  // will contain a copy of each PFCandidate in pfCandidates
  // that remains unmasked. 
  auto_ptr< reco::PFCandidateCollection > 
    pPFCandidateOutput( new reco::PFCandidateCollection ); 
  
  auto_ptr< reco::PFJetCollection > 
    pPFJetOutput( new reco::PFJetCollection ); 
  
  // mask for each PFCandidate.
  // at the beginning, all PFCandidates are unmasked.
  vector<bool> masked( pfCandidates->size(), false);
    
  assert( pfCandidates.isValid() );

  if( pfTaus.isValid() ) {
    const PFTauCollection& taus = *pfTaus;

    if(verbose_) 
      cout<<" PFTaus ------ "<<taus.size()<<endl;
    
    for(unsigned i=0; i<taus.size(); i++) {
      
      
      PFTauRef tauRef(pfTaus, i );
      CandidateBaseRef baseRef( tauRef );
 
      CandidateBaseRefVector ancestors;
      refToAncestor( baseRef,
		     ancestors,
		     pfCandidatesID );
      
      if(verbose_) {
	cout<<"    tau "<<i<<" : "<<taus[i]<<endl;
	printAncestors( ancestors, pfCandidates );
      }
      maskAncestors( ancestors, masked );
    }
  }

  
  if( pfJets.isValid() ) {
    const PFJetCollection& jets = *pfJets;

    if(verbose_) 
      cout<<" PFJets ------ "<<jets.size()<<endl;
    
    for(unsigned i=0; i<jets.size(); i++) {

      PFJetRef jetRef(pfJets, i );
      CandidateBaseRef baseRef( jetRef );

      CandidateBaseRefVector ancestors;
      refToAncestor( baseRef,
		     ancestors,
		     pfCandidatesID );

      if(verbose_) {
	cout<<"    jet "<<i<<endl;
	printAncestors( ancestors, pfCandidates );
      }

      maskAncestors( ancestors, masked );
    }
  }

  

  if( pfPileUpCandidates.isValid() ) {  
    const PileUpPFCandidateCollection& pileUps = *pfPileUpCandidates;
    
    if(verbose_) 
      cout<<" PFPU ------ "<<pileUps.size()<<endl;
 
    for(unsigned i=0; i<pileUps.size(); i++) {
      
      PileUpPFCandidateRef isoRef( pfPileUpCandidates, i ); 
      CandidateBaseRef baseRef( isoRef );

      CandidateBaseRefVector ancestors;
      refToAncestor( baseRef,
		     ancestors,
		     pfCandidatesID );
      if(verbose_) {
	printAncestors( ancestors, pfCandidates );
      }
      maskAncestors( ancestors, masked );

    }
  }
  

  if( pfIsolatedElectrons.isValid() ) {
    const IsolatedPFCandidateCollection& isolated = *pfIsolatedElectrons;

    if(verbose_) 
      cout<<" PFIso ele ------ "<<isolated.size()<<endl;
    
    for(unsigned i=0; i<isolated.size(); i++) {

      IsolatedPFCandidateRef isoRef( pfIsolatedElectrons, i ); 
      CandidateBaseRef baseRef( isoRef );
      CandidateBaseRefVector ancestors;
      refToAncestor( baseRef,
		     ancestors,
		     pfCandidatesID );

      if(verbose_) {
	printAncestors( ancestors, pfCandidates );
      }

      maskAncestors( ancestors, masked );
    }
  }
  

  if( pfIsolatedMuons.isValid() ) {
    const IsolatedPFCandidateCollection& isolated = *pfIsolatedMuons;

    if(verbose_) 
      cout<<" PFIso mu ------ "<<isolated.size()<<endl;
    
    for(unsigned i=0; i<isolated.size(); i++) {

      IsolatedPFCandidateRef isoRef( pfIsolatedMuons, i ); 
      CandidateBaseRef baseRef( isoRef );
      CandidateBaseRefVector ancestors;
      refToAncestor( baseRef,
		     ancestors,
		     pfCandidatesID );

      if(verbose_) {
	printAncestors( ancestors, pfCandidates );
      }

      maskAncestors( ancestors, masked );
    }
  }
  


  const PFCandidateCollection& inCands = *pfCandidates;

  if(verbose_) 
    cout<<" Remaining ------ "<<endl;
  
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
      pPFCandidateOutput->push_back( inCands[i] );
      pPFCandidateOutput->back().setSourceRef(motherRef);
    }
  }
  

  iEvent.put( pPFCandidateOutput );
  iEvent.put( pPFJetOutput );
  
  //   LogDebug("PFTopProjector")<<"STOP event: "<<iEvent.id().event()
  //  			 <<" in run "<<iEvent.id().run()<<endl;
}


void
PFTopProjector::refToAncestor( CandidateBaseRef candRef,
			       CandidateBaseRefVector& ancestors,
			       const edm::ProductID& ancestorsID) const {

  

  //   CandidateBaseRefVector mothers = candRef->motherRefs(); 
 
  unsigned nSources = candRef->numberOfSourceCandidateRefs();

  //   cout<<"going down from "<<candRef.id()
  //       <<"/"<<candRef.key()<<" #mothers "<<nSources
  //       <<" ancestor id "<<allPFCandidates.id()<<endl;
  
  for(unsigned i=0; i<nSources; i++) {
    
    CandidateBaseRef mother = candRef->sourceCandidateRef(i);
    //     cout<<"  mother id "<<mother.id()<<endl;
    
    if(  mother.id() != ancestorsID ) {
      // the mother is not yet at lowest level
      refToAncestor( mother, ancestors, ancestorsID);
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
    
    //     if(verbose_) {
    //       ProductID id = ancestors[i].id();
    //       cout<<"\tmasking "<<index<<", ancestor "<<id<<"/"<<index<<endl;
    //     }
    masked[index] = true;
  }
}



void  PFTopProjector::printAncestors( const reco::CandidateBaseRefVector& ancestors,
				      const edm::Handle<reco::PFCandidateCollection> allPFCandidates ) const {
  
  PFCandidateCollection pfs = *allPFCandidates;

  for(unsigned i=0; i<ancestors.size(); i++) {

    ProductID id = ancestors[i].id();
    assert( id == allPFCandidates.id() );
 
    unsigned index = ancestors[i].key();
    assert( index < pfs.size() );
    
    cout<<"   "<<pfs[index]<<endl;
  }
}



