#include "PhysicsTools/PFCandProducer/interface/PFIsolation.h"
#include "PhysicsTools/PFCandProducer/interface/FetchCollection.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;

PFIsolation::PFIsolation(const edm::ParameterSet& iConfig) {
  


  inputTagPFCandidates_ 
    = iConfig.getParameter<InputTag>("PFCandidates");

  inputTagPFCandidatesForIsolation_ 
    = iConfig.getParameter<InputTag>("PFCandidatesForIsolation");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);



  produces<reco::IsolatedPFCandidateCollection>();
  


  max_ptFraction_InCone_ 
    = iConfig.getParameter<double>("max_ptFraction_InCone");  

  isolation_Cone_DeltaR_
    = iConfig.getParameter<double>("isolation_Cone_DeltaR");  

//   LogDebug("PFIsolation")
//     <<" input collection : "<<inputTagPFCandidates_ <<"\t"
//     <<" max_ptFraction_InCone : "<<max_ptFraction_InCone_<<"\t";
   
  

}



PFIsolation::~PFIsolation() { }



void PFIsolation::beginJob(const edm::EventSetup & es) { }


void PFIsolation::produce(Event& iEvent, 
			  const EventSetup& iSetup) {
  
//   LogDebug("PFIsolation")<<"START event: "<<iEvent.id().event()
// 			 <<" in run "<<iEvent.id().run()<<endl;
  
  
  
  // get PFCandidates

  Handle<PFCandidateCollection> pfCandidates;
  pfpat::fetchCollection(pfCandidates, 
			 inputTagPFCandidates_, 
			 iEvent );

  Handle<PFCandidateCollection> pfCandidatesForIsolation;
  pfpat::fetchCollection( pfCandidatesForIsolation, 
			  inputTagPFCandidatesForIsolation_, 
			  iEvent );

  // get PFCandidates for isolation
  
  
  auto_ptr< reco::IsolatedPFCandidateCollection > 
    pOutput( new reco::IsolatedPFCandidateCollection ); 
  
  for( unsigned i=0; i<pfCandidates->size(); i++ ) {

    const PFCandidateRef candref( pfCandidates,i);

    double ptFractionInCone = computeIsolation( *candref,
						*pfCandidatesForIsolation,
						isolation_Cone_DeltaR_ );

    if( ptFractionInCone < max_ptFraction_InCone_ ) {
      pOutput->push_back( IsolatedPFCandidate( candref, 
					       ptFractionInCone ) );
    }
    
  }
  
  iEvent.put( pOutput );
  
//   LogDebug("PFIsolation")<<"STOP event: "<<iEvent.id().event()
// 			 <<" in run "<<iEvent.id().run()<<endl;
}


double 
PFIsolation::computeIsolation( const PFCandidate& cand,
			       const PFCandidateCollection& candsForIsolation,
			       double isolationCone ) 
  const {

  
  double sumpt = 0;

  if(verbose_) {
    cout<<"compute isolation for "<<cand<<endl;
  }
    
  
  for( unsigned i=0; i<candsForIsolation.size(); i++ ) {
    
    double dR = deltaR( cand,  candsForIsolation[i] );

    // need a clean way to compare candidates for equality... 
    // maybe checking the components as well: same blocks, same elements
    // need to wait 
    if( dR < 1e-12 ) continue;

    if( verbose_ ) {
      cout<<"\t"<<candsForIsolation[i]<<endl;
    }

    if(dR < isolationCone ) {
      if( verbose_ ) {
	cout<<"\t\tpassed ! DeltaR = "<<dR
	    <<", pT = "<<candsForIsolation[i].pt()<<endl;
      }
      sumpt += candsForIsolation[i].pt();
    }
  }
  
  sumpt /= cand.pt();

  if( verbose_ ) {
    cout<<"\tisolation = "<<sumpt<<endl;
  }

  return sumpt;

}


