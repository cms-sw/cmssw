#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"

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

  const PFCandidate& mother = *(c.parent());
  out<<"PileUpPFCandidate, "
     <<mother;
  
  return out;
}
