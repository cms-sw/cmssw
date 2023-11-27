#ifndef L1ScoutingRawToDigi_conversion_h
#define L1ScoutingRawToDigi_conversion_h

#include "EventFilter/L1ScoutingRawToDigi/interface/scales.h"

namespace l1ScoutingRun3 {

  inline float _setPhiRange(float phi) {
    phi = phi >= M_PI ? phi-2.*M_PI : phi;
    return phi;
  }
  
  namespace ugmt {

    inline float fPt(int hwPt) {
      return scales::pt_scale*(hwPt-1);
    };
    inline float fEta(int hwEta) {
      return scales::eta_scale*hwEta;
    };
    inline float fPhi(int hwPhi) {
      //float fPhi_ = scales::phi_scale*hwPhi;
      return _setPhiRange(scales::phi_scale*hwPhi);
    };
    inline float fPtUnconstrained(int hwPtUnconstrained) {
        return scales::ptunconstrained_scale*(hwPtUnconstrained-1);
    };
    inline float fEtaAtVtx(int hwEtaAtVtx) {
      return scales::eta_scale*hwEtaAtVtx;
    };
    inline float fPhiAtVtx(int hwPhiAtVtx) {
      //float fPhi_ = scales::phi_scale*hwPhiAtVtx;
      return _setPhiRange(scales::phi_scale*hwPhiAtVtx);
    };

  } // namespace ugmt

  namespace demux {

    inline float fEt(int hwEt) {
      return scales::et_scale*hwEt;
    };
    inline float fEta(int hwEta) {
      return scales::eta_scale*hwEta;
    };
    inline float fPhi(int hwPhi) {
      //float fPhi_ = scales::phi_scale*hwPhi;
      return _setPhiRange(scales::phi_scale*hwPhi);
    };

  } // namespace demux
  
} // namespace l1ScoutingRun3


#endif