#ifndef TkTrigger_L1Em_h
#define TkTrigger_L1Em_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {

  class TkEm : public L1Candidate {
  public:
    TkEm();

    TkEm(const LorentzVector& p4, const edm::Ref<EGammaBxCollection>& egRef, float tkisol = -999.);

    TkEm(const LorentzVector& p4,
                   const edm::Ref<EGammaBxCollection>& egRef,
                   float tkisol = -999.,
                   float tkisolPV = -999);

    virtual ~TkEm() {}

    // ---------- const member functions ---------------------

    const edm::Ref<EGammaBxCollection>& EGRef() const { return egRef_; }

    const double l1RefEta() const { return egRef_->eta(); }

    const double l1RefPhi() const { return egRef_->phi(); }

    const double l1RefEt() const { return egRef_->et(); }

    float trkIsol() const { return TrkIsol_; }  // not constrained to the PV, just track ptSum

    float trkIsolPV() const { return TrkIsolPV_; }  // constrained to the PV by DZ

    // ---------- member functions ---------------------------

    void setTrkIsol(float TrkIsol) { TrkIsol_ = TrkIsol; }
    void setTrkIsolPV(float TrkIsolPV) { TrkIsolPV_ = TrkIsolPV; }

    //	 int bx() const;

  private:
    edm::Ref<EGammaBxCollection> egRef_;
    float TrkIsol_;
    float TrkIsolPV_;
  };
}  // namespace l1t

#endif
