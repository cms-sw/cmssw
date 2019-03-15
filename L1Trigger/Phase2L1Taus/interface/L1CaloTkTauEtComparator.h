#ifndef L1CaloTkTauEtComparator_HH
#define L1CaloTkTauEtComparator_HH
#include "DataFormats/L1TrackTrigger/interface/L1CaloTkTauParticle.h"

namespace L1CaloTkTau{
  class EtComparator {
  public:
    bool operator()(const l1t::L1CaloTkTauParticle& a, const l1t::L1CaloTkTauParticle& b) const {
      double et_a = a.et();
      double et_b = b.et();
      return et_a > et_b;
    }
  };

  class PtComparator {
  public:
    bool operator()(const l1t::L1CaloTkTauParticle& a, const l1t::L1CaloTkTauParticle& b) const {
      double pt_a = a.getTrackBasedP4().Pt();
      double pt_b = b.getTrackBasedP4().Pt();
      return pt_a > pt_b;
    }
  };

}
#endif

