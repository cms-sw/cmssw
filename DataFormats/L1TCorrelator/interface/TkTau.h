#ifndef DataFormatsL1TCorrelator_TkTau_h
#define DataFormatsL1TCorrelator_TkTau_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkTau
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1TCorrelator/interface/TkEm.h"

#include "DataFormats/L1Trigger/interface/Tau.h"

namespace l1t {

  class TkTau : public L1Candidate {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef std::vector<L1TTTrackType> L1TTTrackCollection;

    TkTau();

    TkTau(const LorentzVector& p4,
          const edm::Ref<TauBxCollection>& tauCaloRef,  // null for stand-alone TkTaus
          const edm::Ptr<L1TTTrackType>& trkPtr,
          const edm::Ptr<L1TTTrackType>& trkPtr2,  // null for tau -> 1 prong
          const edm::Ptr<L1TTTrackType>& trkPtr3,  // null for tau -> 1 prong
          float tkisol = -999.);

    // ---------- const member functions ---------------------

    const edm::Ref<TauBxCollection>& tauCaloRef() const { return tauCaloRef_; }

    const edm::Ptr<L1TTTrackType>& trkPtr() const { return trkPtr_; }

    const edm::Ptr<L1TTTrackType>& trkPtr2() const { return trkPtr2_; }
    const edm::Ptr<L1TTTrackType>& trkPtr3() const { return trkPtr3_; }

    float trkzVtx() const { return TrkzVtx_; }
    float trkIsol() const { return TrkIsol_; }

    // ---------- member functions ---------------------------

    void setTrkzVtx(float TrkzVtx) { TrkzVtx_ = TrkzVtx; }
    void setTrkIsol(float TrkIsol) { TrkIsol_ = TrkIsol; }
    int bx() const;

  private:
    edm::Ref<TauBxCollection> tauCaloRef_;

    edm::Ptr<L1TTTrackType> trkPtr_;
    edm::Ptr<L1TTTrackType> trkPtr2_;
    edm::Ptr<L1TTTrackType> trkPtr3_;

    float TrkIsol_;
    float TrkzVtx_;
  };
}  // namespace l1t

#endif
