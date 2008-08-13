#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

using namespace reco;

PileUpPFCandidate::PileUpPFCandidate() : PFCandidate() {}


PileUpPFCandidate::PileUpPFCandidate( const PFCandidateRef & candidateRef ) : 
  PFCandidate(candidateRef) {

}

PileUpPFCandidate * PileUpPFCandidate::clone() const {
  return new PileUpPFCandidate( * this );
}


PileUpPFCandidate::~PileUpPFCandidate();


std::ostream& operator<<( std::ostream& out, 
			  const PileUpPFCandidate& c ) {
  if(!out) return out;
  
  
  out<<"PileUpPFCandidate, "
     <<c.sourceCandidateRef(0).id()<<"/"
     <<c.sourceCandidateRef(0).key();
  
  return out;
}
