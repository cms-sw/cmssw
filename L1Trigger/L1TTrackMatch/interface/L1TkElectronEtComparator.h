#ifndef L1Trigger_L1TTrackMatch_L1TkElectronEtComparator_HH
#define L1Trigger_L1TTrackMatch_L1TkElectronEtComparator_HH
#include "DataFormats/L1Trigger/interface/EGamma.h"

namespace L1TkElectron {
  class EtComparator {
  public:
    bool operator()(const l1t::EGamma& a, const l1t::EGamma& b) const {
      double et_a = 0.0;
      double et_b = 0.0;
      double cosh_a_eta = cosh(a.eta());
      double cosh_b_eta = cosh(b.eta());

      if (cosh_a_eta > 0.0)
        et_a = a.energy() / cosh_a_eta;
      if (cosh_b_eta > 0.0)
        et_b = b.energy() / cosh_b_eta;

      return et_a > et_b;
    }
  };
}  // namespace L1TkElectron
#endif
