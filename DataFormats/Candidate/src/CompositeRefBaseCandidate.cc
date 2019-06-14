#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace reco;

CompositeRefBaseCandidate::~CompositeRefBaseCandidate() {}

CompositeRefBaseCandidate* CompositeRefBaseCandidate::clone() const { return new CompositeRefBaseCandidate(*this); }

const Candidate* CompositeRefBaseCandidate::daughter(size_type i) const {
  return (i < numberOfDaughters()) ? &*dau[i] : nullptr;  // i >= 0, since i is unsigned
}

const Candidate* CompositeRefBaseCandidate::mother(size_type i) const { return nullptr; }

Candidate* CompositeRefBaseCandidate::daughter(size_type i) { return nullptr; }

size_t CompositeRefBaseCandidate::numberOfDaughters() const { return dau.size(); }

size_t CompositeRefBaseCandidate::numberOfMothers() const { return 0; }

bool CompositeRefBaseCandidate::overlap(const Candidate& c2) const {
  throw cms::Exception("Error") << "can't check overlap internally for CompositeRefBaseCanddate";
}
