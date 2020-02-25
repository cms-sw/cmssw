#ifndef L1TrackTrigger_L1TkBsCandidate_h
#define L1TrackTrigger_L1TkBsCandidate_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TrackerTrigger
// Class:       L1TkBsCandidate
// 

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidateFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {         
  class L1TkBsCandidate: public L1Candidate {
  public:
    
    L1TkBsCandidate();
    L1TkBsCandidate(const LorentzVector& p4,
		    L1TkPhiCandidate cand1,
		    L1TkPhiCandidate cand2);
    
    virtual ~L1TkBsCandidate() {}
    
    // ---------- const member functions ---------------------    
    const L1TkPhiCandidate& getPhiCandidate(size_t i) const { return phiCandList_.at(i); }
    
    // ---------- member functions ---------------------------

    // deltaR between track pair
    double dRPhiPair() const;

    // position difference between track pair
    double dxyPhiPair() const;
    double dzPhiPair() const;

  private:
    
    L1TkPhiCandidateCollection phiCandList_;
  };
}
#endif
