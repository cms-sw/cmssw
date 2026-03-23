#ifndef L1TriggerScouting_Utilities_conversion_h
#define L1TriggerScouting_Utilities_conversion_h

#include "L1TriggerScouting/Utilities/interface/scales.h"
#include <cstdint>

namespace l1ScoutingRun3 {

  inline float _setPhiRange(float phi) {
    phi = phi >= M_PI ? phi - 2. * M_PI : phi;
    return phi;
  }

  namespace ugmt {

    inline float fPt(int hwPt) { return scales::pt_scale * (hwPt - 1); };
    inline float fEta(int hwEta) { return scales::eta_scale * hwEta; };
    inline float fPhi(int hwPhi) { return _setPhiRange(scales::phi_scale * hwPhi); };
    inline float fPtUnconstrained(int hwPtUnconstrained) {
      return scales::ptunconstrained_scale * (hwPtUnconstrained - 1);
    };
    inline float fEtaAtVtx(int hwEtaAtVtx) { return scales::eta_scale * hwEtaAtVtx; };
    inline float fPhiAtVtx(int hwPhiAtVtx) { return _setPhiRange(scales::phi_scale * hwPhiAtVtx); };

  }  // namespace ugmt

  namespace demux {

    inline float fEt(int hwEt) { return scales::et_scale * hwEt; };
    inline float fEta(int hwEta) { return scales::eta_scale * hwEta; };
    inline float fPhi(int hwPhi) { return _setPhiRange(scales::phi_scale * hwPhi); };

  }  // namespace demux

  namespace calol1 {

    inline float fEt(int16_t hwEt) { return scales::et_scale * hwEt; };

    inline constexpr int16_t kHwEtaAbsMin = 1;
    inline constexpr int16_t kHwEtaAbsMax = 41;
    inline constexpr int16_t kHwEtaAbsHFFirst = 29;

    bool validHwEta(int16_t hwEta);
    float fEta(int16_t hwEta);

    inline constexpr int16_t kHwPhiMin = 1;
    inline constexpr int16_t kHwPhiMax = 72;

    bool validHwPhi(int16_t hwPhi);
    float fPhi(int16_t hwPhi);
  }  // namespace calol1

}  // namespace l1ScoutingRun3

#endif
