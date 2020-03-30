#ifndef L1TrackTrigger_TkBsCandidateFwd_h
#define L1TrackTrigger_TkBsCandidateFwd_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TrackTrigger
// File:        TkBsCandidateFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class TkBsCandidate;

  using TkBsCandidateCollection = std::vector<TkBsCandidate>;
  using TkBsCandidateRef = edm::Ref<TkBsCandidateCollection>;
  using TkBsCandidateRefVector = edm::RefVector<TkBsCandidateCollection>;
  using TkBsCandidateVectorRef = std::vector<TkBsCandidateRef>;
}  // namespace l1t
#endif
