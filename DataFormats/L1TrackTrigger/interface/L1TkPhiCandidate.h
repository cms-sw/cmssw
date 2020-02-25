#ifndef L1TrackTrigger_L1TkPhiCandidate_h
#define L1TrackTrigger_L1TkPhiCandidate_h

// -*- C++ -*-
//
// Package:     DataFormats/L1TrackTrigger
// Class:       L1TkPhiCandidate
// 

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {         

  class L1TkPhiCandidate: public L1Candidate {
  public:
    
    static constexpr double kmass = 0.493; // GeV
    static constexpr double phi_polemass = 1.019445; // GeV

    using L1TTTrackType = TTTrack<Ref_Phase2TrackerDigi_>;
    using L1TTTrackCollection = std::vector<L1TTTrackType>;
    
    L1TkPhiCandidate();
    L1TkPhiCandidate(const LorentzVector& p4,
                     const edm::Ptr<L1TTTrackType>& trkPtr1,
                     const edm::Ptr<L1TTTrackType>& trkPtr2);
    
    virtual ~L1TkPhiCandidate() {}
    
    // ---------- const member functions ---------------------    
    const edm::Ptr<L1TTTrackType>& getTrkPtr(size_t i) const { return trkPtrList_.at(i); }

    // ---------- member functions ---------------------------

    // deltaR between track pair
    double dRTrkPair() const;

    // difference from nominal mass
    double dmass() const;

    // position difference between track pair
    double dxyTrkPair() const;
    double dzTrkPair() const;

    double vx() const override;
    double vy() const override;
    double vz() const override;

  private:
    
    std::vector<edm::Ptr<L1TTTrackType>> trkPtrList_;
  };
}
#endif
