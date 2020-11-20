#include "Rtypes.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"

namespace DataFormats_L1TParticleFlow {
  struct dictionary {
    l1t::HPSPFTau hpspftau;
    l1t::HPSPFTauCollection hpspftauCollection;
    edm::Wrapper<l1t::HPSPFTauCollection> hpspftauCWrapper;
    l1t::HPSPFTauRefVector hpspftaureRefVector;
    l1t::HPSPFTauVectorRef hpspftauVectorRef;
  };
}  // namespace DataFormats_L1TParticleFlow
