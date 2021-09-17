#ifndef DataFormats_L1TParticleFlow_HPSPFTauFwd_H
#define DataFormats_L1TParticleFlow_HPSPFTauFwd_H

#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"

namespace l1t {
  typedef std::vector<HPSPFTau> HPSPFTauCollection;

  typedef edm::Ref<HPSPFTauCollection> HPSPFTauRef;
  typedef edm::RefVector<HPSPFTauCollection> HPSPFTauRefVector;
  typedef std::vector<HPSPFTauRef> HPSPFTauVectorRef;
}  // namespace l1t

#endif
