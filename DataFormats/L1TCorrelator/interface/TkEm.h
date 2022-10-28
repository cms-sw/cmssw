#ifndef DataFormatsL1TCorrelator_TkEm_h
#define DataFormatsL1TCorrelator_TkEm_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <ap_int.h>

namespace l1t {

  class TkEm : public L1Candidate {
  public:
    TkEm();

    TkEm(const LorentzVector& p4, const edm::Ref<EGammaBxCollection>& egRef, float tkisol = -999.);

    TkEm(const LorentzVector& p4,
         const edm::Ref<EGammaBxCollection>& egRef,
         float tkisol = -999.,
         float tkisolPV = -999);

    // ---------- const member functions ---------------------

    const edm::Ref<EGammaBxCollection>& EGRef() const { return egRef_; }

    const double l1RefEta() const { return egRef_->eta(); }

    const double l1RefPhi() const { return egRef_->phi(); }

    const double l1RefEt() const { return egRef_->et(); }

    float trkIsol() const { return trkIsol_; }          // not constrained to the PV, just track ptSum
    float trkIsolPV() const { return trkIsolPV_; }      // constrained to the PV by DZ
    float pfIsol() const { return pfIsol_; }            // not constrained to the PV, just track ptSum
    float pfIsolPV() const { return pfIsolPV_; }        // constrained to the PV by DZ
    float puppiIsol() const { return puppiIsol_; }      // not constrained to the PV, just track ptSum
    float puppiIsolPV() const { return puppiIsolPV_; }  // constrained to the PV by DZ

    // ---------- member functions ---------------------------

    void setTrkIsol(float TrkIsol) { trkIsol_ = TrkIsol; }
    void setTrkIsolPV(float TrkIsolPV) { trkIsolPV_ = TrkIsolPV; }
    void setPFIsol(float pfIsol) { pfIsol_ = pfIsol; }
    void setPFIsolPV(float pfIsolPV) { pfIsolPV_ = pfIsolPV; }
    void setPuppiIsol(float puppiIsol) { puppiIsol_ = puppiIsol; }
    void setPuppiIsolPV(float puppiIsolPV) { puppiIsolPV_ = puppiIsolPV; }
    void setEGRef(const edm::Ref<EGammaBxCollection>& egRef) { egRef_ = egRef; }

    template <int N>
    void setEgBinaryWord(ap_uint<N> word) {
      egBinaryWord0_ = word;
      egBinaryWord1_ = (word >> 32);
      egBinaryWord2_ = (word >> 64);
    }

    template <int N>
    ap_uint<N> egBinaryWord() const {
      return ap_uint<N>(egBinaryWord0_) | (ap_uint<N>(egBinaryWord1_) << 32) | (ap_uint<N>(egBinaryWord2_) << 64);
    }

  private:
    edm::Ref<EGammaBxCollection> egRef_;
    float trkIsol_;
    float trkIsolPV_;
    float pfIsol_;
    float pfIsolPV_;
    float puppiIsol_;
    float puppiIsolPV_;
    uint32_t egBinaryWord0_;
    uint32_t egBinaryWord1_;
    uint32_t egBinaryWord2_;
  };
}  // namespace l1t

#endif
