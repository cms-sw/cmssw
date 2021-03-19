#ifndef DataFormats_L1TParticleFlow_HPSPFTauFwd_H
#define DataFormats_L1TParticleFlow_HPSPFTauFwd_H

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {

  class HPSPFTau;

  typedef std::vector<HPSPFTau> HPSPFTauCollection;

  typedef edm::Ref<HPSPFTauCollection> HPSPFTauRef;
  typedef edm::RefVector<HPSPFTauCollection> HPSPFTauRefVector;
  typedef std::vector<HPSPFTauRef> HPSPFTauVectorRef;
}  // namespace l1t

#endif
