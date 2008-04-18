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
  
  
  const CandidateBaseRefVector mothers = c.motherRefs();
  assert(mothers.size()==1);
  out<<"PileUpPFCandidate, "
     <<mothers[0].id()<<"/"<<mothers[0].key();
  
  return out;
}
