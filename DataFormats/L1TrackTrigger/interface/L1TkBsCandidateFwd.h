#ifndef L1TrackTrigger_L1TkBsCandidateFwd_h
#define L1TrackTrigger_L1TkBsCandidateFwd_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TrackTrigger
// File:        L1TkBsCandidateFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class L1TkBsCandidate;
  
  using L1TkBsCandidateCollection = std::vector<L1TkBsCandidate>;
  using L1TkBsCandidateRef        = edm::Ref<L1TkBsCandidateCollection>;
  using L1TkBsCandidateRefVector  = edm::RefVector<L1TkBsCandidateCollection>;
  using L1TkBsCandidateVectorRef  = std::vector<L1TkBsCandidateRef>;
}
#endif
