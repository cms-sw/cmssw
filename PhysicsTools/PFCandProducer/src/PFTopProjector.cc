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

//TODO just for testing, remove this
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;

const char* PFTopProjector::pfJetsOutLabel_ = "PFJets";
const char* PFTopProjector::pfCandidatesOutLabel_ = "PFCandidates";

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


  // produces<reco::PFCandidateCollection>("PFCandidates");
  //  produces<reco::PFJetCollection>("PFJets");

  produces<reco::PFCandidateCollection>(pfCandidatesOutLabel_); 
  produces<reco::PFJetCollection>(pfJetsOutLabel_);
  
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
  
  
  if( verbose_)
    cout<<"Event -------------------- "<<iEvent.id().event()<<endl;
  
  // get the various collections

  // this collection is the collection of PFCandidates to 
  // be masked by the other ones
  Handle<PFCandidateCollection> pfCandidates;
  pfpat::fetchCollection(pfCandidates, 
			 inputTagPFCandidates_, 
			 iEvent );

  if( !pfCandidates.isValid() ) {
    std::ostringstream  err;
    err<<"The collection of input PFCandidates must be supplied."<<endl
       <<"It is now set to : "<<inputTagPFCandidates_<<endl;
    edm::LogError("PFPAT")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }

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

  Handle<PFCandidateCollection> pfIsolatedElectrons;
  pfpat::fetchCollection(pfIsolatedElectrons, 
			 inputTagIsolatedElectrons_, 
			 iEvent );

  Handle<PFCandidateCollection> pfIsolatedMuons;
  pfpat::fetchCollection(pfIsolatedMuons, 
			 inputTagIsolatedMuons_, 
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


  processCollection( pfTaus, pfCandidates, masked, 
		     "PFTau");
  processCollection( pfJets, pfCandidates, masked, 
		     "PFJet");
  processCollection( pfPileUpCandidates, pfCandidates, masked, 
		     "PileUpParticle");
  processCollection( pfIsolatedElectrons, pfCandidates, masked, 
		     "IsoElectron");
  processCollection( pfIsolatedMuons, pfCandidates, masked, 
		     "IsoMuon");


  const PFCandidateCollection& inCands = *pfCandidates;

  if(verbose_) 
    cout<<" Remaining PFCandidates ------ "<<endl;
  
  for(unsigned i=0; i<inCands.size(); i++) {
    
    if(masked[i]) {
      if(verbose_)
	cout<<"X "<<i<<" "<<inCands[i]<<endl;
      continue;
    }
    else {
      if(verbose_)
	cout<<"O "<<i<<" "<<inCands[i]<<endl;
      PFCandidatePtr motherPtr( pfCandidates, i );
      pPFCandidateOutput->push_back( inCands[i] );
      pPFCandidateOutput->back().setSourcePtr(motherPtr);
    }
  }

  iEvent.put( pPFCandidateOutput, pfCandidatesOutLabel_ );


  // now mask the jets with the taus (if the jet collection has been provided)
  
  if( pfJets.isValid() ) {
    vector<bool> maskedJets( pfJets->size(), false);

    
    processCollection( pfTaus, pfJets, maskedJets, 
		       "PFTau masking PFJets");


    const PFJetCollection& inJets = *pfJets;
    
    if(verbose_) 
      cout<<" Remaining PFJets ------ "<<endl;
    
    for(unsigned i=0; i<inJets.size(); i++) {
      
      if(maskedJets[i]) {
	if(verbose_)
	  cout<<"X "<<i<<" "<<inJets[i]<<endl;
	continue;
      }
      else {
	if(verbose_)
	  cout<<"O "<<i<<" "<<inJets[i]<<endl;
	pPFJetOutput->push_back( inJets[i] );
      }
    }
  }



  iEvent.put( pPFJetOutput, pfJetsOutLabel_);
  
  //   LogDebug("PFTopProjector")<<"STOP event: "<<iEvent.id().event()
  //  			 <<" in run "<<iEvent.id().run()<<endl;
}


void
  PFTopProjector::ptrToAncestor( reco::CandidatePtr candPtr,
				 reco::CandidatePtrVector& ancestors,
				 const edm::ProductID& ancestorsID ) const {

  
 
  unsigned nSources = candPtr->numberOfSourceCandidatePtrs();

//   cout<<"going down from "<<candPtr.id()
//       <<"/"<<candPtr.key()<<" #mothers "<<nSources
//       <<" ancestor id "<<ancestorsID<<endl;
  
  for(unsigned i=0; i<nSources; i++) {
    
    CandidatePtr mother = candPtr->sourceCandidatePtr(i);
//     cout<<"  mother id "<<mother.id()<<endl;
    
    if(  mother.id() != ancestorsID ) {
      // the mother is not yet at lowest level
      ptrToAncestor( mother, ancestors, ancestorsID);
    }
    else {
      // adding mother to the list of ancestors
      ancestors.push_back( mother ); 
    }
  }
}



void PFTopProjector::maskAncestors( const reco::CandidatePtrVector& ancestors,
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



// void  PFTopProjector::printAncestors( const reco::CandidatePtrVector& ancestors,
// 				      const edm::Handle<reco::PFCandidateCollection>& allPFCandidates ) const {
  
//   PFCandidateCollection pfs = *allPFCandidates;

//   for(unsigned i=0; i<ancestors.size(); i++) {

//     ProductID id = ancestors[i].id();
//     assert( id == allPFCandidates.id() );
 
//     unsigned index = ancestors[i].key();
//     assert( index < pfs.size() );
    
//     cout<<"\t\t"<<pfs[index]<<endl;
//   }
// }

