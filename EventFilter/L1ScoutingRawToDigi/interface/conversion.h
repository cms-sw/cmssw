#ifndef L1ScoutingRawToDigi_conversion_h
#define L1ScoutingRawToDigi_conversion_h

#include "EventFilter/L1ScoutingRawToDigi/interface/scales.h"

namespace scoutingRun3 {
  
  namespace ugmt {

    inline float fPt(int hwPt) {
      return scales::pt_scale*(hwPt-1);
    };
    inline float fEta(int hwEta) {
      return scales::eta_scale*hwEta;
    };
    inline float fPhi(int hwPhi) {
      float fPhi_ = scales::phi_scale*hwPhi;
      fPhi_ = fPhi_>=M_PI ? fPhi_-2.*M_PI : fPhi_;
      return fPhi_;
    };
    inline float fPtUnconstrained(int hwPtUnconstrained) {
        return scales::ptunconstrained_scale*hwPtUnconstrained;
    };
    inline float fEtaAtVtx(int hwEtaAtVtx) {
      return scales::eta_scale*hwEtaAtVtx;
    };
    inline float fPhiAtVtx(int hwPhiAtVtx) {
      float fPhi_ = scales::phi_scale*hwPhiAtVtx;
      fPhi_ = fPhi_>=M_PI ? fPhi_-2.*M_PI : fPhi_;
      return fPhi_;
    };

  } // namespace ugmt

  namespace demux {

    inline float fEt(int hwEt) {
      return scales::et_scale*(hwEt-1);
    };
    inline float fEta(int hwEta) {
      return scales::eta_scale*hwEta;
    };
    inline float fPhi(int hwPhi) {
      float fPhi_ = scales::phi_scale*hwPhi;
      fPhi_ = fPhi_>= M_PI ? fPhi_-2.*M_PI : fPhi_;
      return fPhi_;
    };

  } // namespace demux
  
} // namespace scoutingRun3


#endif