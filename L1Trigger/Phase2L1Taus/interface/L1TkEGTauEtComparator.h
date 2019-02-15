#ifndef L1TkEGEtComparator_HH
#define L1TkEGEtComparator_HH
#include "DataFormats/L1TrackTrigger/interface/L1TkEGTauParticle.h"

namespace L1TkEG{
  class EtComparator {
  public:
    bool operator()(const l1t::L1TkEGTauParticle& a, const l1t::L1TkEGTauParticle& b) const {
      double et_a = a.et();
      double et_b = b.et();
      return et_a > et_b;
    }
  };

  class PtComparator {
  public:
    bool operator()(const l1t::L1TkEGTauParticle& a, const l1t::L1TkEGTauParticle& b) const {
      double et_a = a.pt();
      double et_b = b.pt();
      return et_a > et_b;
    }
  };

}
#endif

