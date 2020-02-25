#include "Rtypes.h"

#include "DataFormats/Phase2L1Taus/interface/L1HPSPFTau.h"
#include "DataFormats/Phase2L1Taus/interface/L1HPSPFTauFwd.h"

namespace DataFormats_Phase2L1Taus
{
  struct dictionary 
  {
    l1t::L1HPSPFTau l1hpspftau;
    l1t::L1HPSPFTauCollection l1hpspftauCollection;
    edm::Wrapper<l1t::L1HPSPFTauCollection> l1hpspftauCWrapper;
  };
}
