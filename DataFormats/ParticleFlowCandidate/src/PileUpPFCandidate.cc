#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

using namespace reco;

PileUpPFCandidate::PileUpPFCandidate() : PFCandidate() {}


PileUpPFCandidate::PileUpPFCandidate( const PFCandidatePtr & candidatePtr ) : 
  PFCandidate(candidatePtr) {
}

PileUpPFCandidate * PileUpPFCandidate::clone() const {
  return new PileUpPFCandidate( * this );
}


PileUpPFCandidate::~PileUpPFCandidate();


std::ostream& operator<<( std::ostream& out, 
			  const PileUpPFCandidate& c ) {
  if(!out) return out;
  
  
  out<<"PileUpPFCandidate, "
     <<c.sourceCandidatePtr(0).id()<<"/"
     <<c.sourceCandidatePtr(0).key();
  
  return out;
}
