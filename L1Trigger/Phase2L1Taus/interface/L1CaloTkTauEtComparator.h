#ifndef L1CaloTkTauEtComparator_HH
#define L1CaloTkTauEtComparator_HH
#include "DataFormats/L1TrackTrigger/interface/L1CaloTkTauParticle.h"

namespace L1CaloTkTau{
  class EtComparator {
  public:
    bool operator()(const l1t::L1CaloTkTauParticle& a, const l1t::L1CaloTkTauParticle& b) const {
      double et_a = a.getEt();
      double et_b = b.getEt();
      return et_a > et_b;
    }
  };

  class PtComparator {
  public:
    bool operator()(const l1t::L1CaloTkTauParticle& a, const l1t::L1CaloTkTauParticle& b) const {
      double pt_a = a.pt();
      double pt_b = b.pt();
      return pt_a > pt_b;
    }
  };

}
#endif

