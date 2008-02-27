#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"

using namespace reco;

IsolatedPFCandidate::IsolatedPFCandidate() : PFCandidate(), isolation_(-1) {}


IsolatedPFCandidate::IsolatedPFCandidate( const PFCandidateRef & candidateRef, 
					  double isolation ) : 
  PFCandidate(candidateRef), 
  isolation_(isolation) {

//   parent_ = candidateRef;
}

IsolatedPFCandidate * IsolatedPFCandidate::clone() const {
  return new IsolatedPFCandidate( * this );
}


IsolatedPFCandidate::~IsolatedPFCandidate();


std::ostream& operator<<( std::ostream& out, 
			  const IsolatedPFCandidate& c ) {
  if(!out) return out;

  const PFCandidate& mother = c;
  out<<"IsolatedPFCandidate, isolation = "
     <<c.isolation()<<" "
     <<mother;
  return out;
}
