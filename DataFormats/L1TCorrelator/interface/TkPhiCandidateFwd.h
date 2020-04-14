#ifndef DataFormatsL1TCorrelator_TkPhiCandidateFwd_h
#define DataFormatsL1TCorrelator_TkPhiCandidateFwd_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TCorrelator
// Class  :     TkPhiCandidateFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class TkPhiCandidate;

  using TkPhiCandidateCollection = std::vector<TkPhiCandidate>;
  using TkPhiCandidateRef = edm::Ref<TkPhiCandidateCollection>;
  using TkPhiCandidateRefVector = edm::RefVector<TkPhiCandidateCollection>;
  using TkPhiCandidateVectorRef = std::vector<TkPhiCandidateRef>;
}  // namespace l1t
#endif
