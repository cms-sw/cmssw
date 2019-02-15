#ifndef L1TrkTauEtComparator_HH
#define L1TrkTauEtComparator_HH
#include "DataFormats/L1TrackTrigger/interface/L1TrkTauParticle.h"

namespace L1TrkTau{
  class EtComparator {
  public:
    bool operator()(const l1t::L1TrkTauParticle& a, const l1t::L1TrkTauParticle& b) const {
      double et_a = a.et();
      double et_b = b.et();
      return et_a > et_b;
    }
  };

  class PtComparator {
  public:
    bool operator()(const l1t::L1TrkTauParticle& a, const l1t::L1TrkTauParticle& b) const {
      double et_a = a.pt();
      double et_b = b.pt();
      return et_a > et_b;
    }
  };

}
#endif

