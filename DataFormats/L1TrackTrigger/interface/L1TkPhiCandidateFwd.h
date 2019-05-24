#ifndef L1TrackTrigger_L1TkPhiCandidateFwd_h
#define L1TrackTrigger_L1TkPhiCandidateFwd_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TrackTrigger
// Class  :     L1TkPhiCandidateFwd
// 

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class L1TkPhiCandidate;
  
  using L1TkPhiCandidateCollection = std::vector<L1TkPhiCandidate>;
  using L1TkPhiCandidateRef        = edm::Ref<L1TkPhiCandidateCollection>;
  using L1TkPhiCandidateRefVector  = edm::RefVector<L1TkPhiCandidateCollection>;
  using L1TkPhiCandidateVectorRef  = std::vector<L1TkPhiCandidateRef>;
}
#endif
