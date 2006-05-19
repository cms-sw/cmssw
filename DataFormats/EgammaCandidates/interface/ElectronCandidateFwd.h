#ifndef EgammaReco_ElectronCandidateFwd_h
#define EgammaReco_ElectronCandidateFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ElectronCandidate;

  /// collectin of ElectronCandidate objects
  typedef std::vector<ElectronCandidate> ElectronCandidateCollection;

  /// reference to an object in a collection of ElectronCandidate objects
  typedef edm::Ref<ElectronCandidateCollection> ElectronCandidateRef;

  /// reference to a collection of ElectronCandidate objects
  typedef edm::RefProd<ElectronCandidateCollection> ElectronCandidateRefProd;

  /// vector of objects in the same collection of ElectronCandidate objects
  typedef edm::RefVector<ElectronCandidateCollection> ElectronCandidateRefVector;

  /// iterator over a vector of reference to ElectronCandidate objects
  typedef ElectronCandidateRefVector::iterator electronCandidate_iterator;
}

#endif
