#ifndef DataFormatsL1TCorrelator_TkBsCandidate_h
#define DataFormatsL1TCorrelator_TkBsCandidate_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TCorrelator
// Class:       TkBsCandidate
//

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidate.h"
#include "DataFormats/L1TCorrelator/interface/TkPhiCandidateFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {
  class TkBsCandidate : public L1Candidate {
  public:
    TkBsCandidate();
    TkBsCandidate(const LorentzVector& p4, TkPhiCandidate cand1, TkPhiCandidate cand2);

    // ---------- const member functions ---------------------
    const TkPhiCandidate& phiCandidate(size_t i) const { return phiCandList_.at(i); }

    // ---------- member functions ---------------------------

    // deltaR between track pair
    double dRPhiPair() const;

    // position difference between track pair
    double dxyPhiPair() const;
    double dzPhiPair() const;

  private:
    TkPhiCandidateCollection phiCandList_;
  };
}  // namespace l1t
#endif
