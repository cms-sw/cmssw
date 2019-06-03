#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"

#include <iostream>

using namespace reco;
using namespace std;

IsolatedPFCandidate::IsolatedPFCandidate() : PFCandidate(), isolation_(-1) {}

IsolatedPFCandidate::IsolatedPFCandidate(const PFCandidatePtr& candidatePtr, double isolation)
    : PFCandidate(candidatePtr), isolation_(isolation) {}

IsolatedPFCandidate* IsolatedPFCandidate::clone() const { return new IsolatedPFCandidate(*this); }

IsolatedPFCandidate::~IsolatedPFCandidate() {}

std::ostream& reco::operator<<(std::ostream& out, const IsolatedPFCandidate& c) {
  if (!out)
    return out;

  const PFCandidate& mother = c;
  out << "IsolatedPFCandidate, isolation = " << c.isolation() << " " << mother;
  return out;
}
