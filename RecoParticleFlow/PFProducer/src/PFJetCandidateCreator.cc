// produces a collection
// of candidates <edm::OwnVector<Candidate> from PFCandidates
// author: Joanna Weng, ETH Zurich

#include <cmath>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoParticleFlow/PFProducer/interface/PFJetCandidateCreator.h"
using namespace edm;
using namespace reco;
using namespace std;



PFJetCandidateCreator::PFJetCandidateCreator( const ParameterSet & p ) 
  :
  mVerbose (p.getUntrackedParameter<bool> ("verbose", 0)),
  mSource (p.getParameter<edm::InputTag> ("src")){
  produces<CandidateCollection>();
}


PFJetCandidateCreator::~PFJetCandidateCreator() {
}



void  PFJetCandidateCreator::produce( Event& evt, const EventSetup& ) {
  Handle <PFCandidateCollection> candidates;
  evt.getByLabel( mSource, candidates );
  
  if (mVerbose >= 2){
    std::cout << " PFJetCandidateCreator: producing Candidates from PFCandidate  " << std::endl; 
  }
  
  std::auto_ptr<edm::OwnVector<Candidate> > pCopy( new edm::OwnVector<Candidate> );

  // Copy PFCandidates into edm::OwnVector<Candidate> format 
  // as input for jet algorithms
  for( std::vector<PFCandidate>::const_iterator it = 
	 (*candidates).begin (); it != (*candidates).end(); ++it) {
                        pCopy->push_back( it->clone() );
         }
  evt.put( pCopy);
}
