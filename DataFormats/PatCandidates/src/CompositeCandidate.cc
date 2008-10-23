//
// $Id: CompositeCandidate.cc,v 1.1 2008/01/15 12:59:32 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"


using namespace pat;


/// default constructor
CompositeCandidate::CompositeCandidate() : 
  PATObject<CompositeCandidateType>(CompositeCandidateType(0, CompositeCandidateType::LorentzVector(0, 0, 0, 0), CompositeCandidateType::Point(0,0,0))) {
}


/// constructor from CompositeCandidateType
CompositeCandidate::CompositeCandidate(const CompositeCandidateType & aCompositeCandidate) : 
  PATObject<CompositeCandidateType>(aCompositeCandidate) {
}


/// destructor
CompositeCandidate::~CompositeCandidate() {
}
