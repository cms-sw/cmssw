#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace { 
struct dictionary {
  // L1MuRegionalTriggers -> GMT
  std::vector<L1MuRegionalCand> dummy0;
  edm::Wrapper<std::vector<L1MuRegionalCand> > dummy1;

  // GMT -> GT
  std::vector<L1MuGMTCand> dummy2;
  edm::Wrapper<std::vector<L1MuGMTCand> > dummy3;

  // GMT readout
  L1MuGMTReadoutCollection dummy4;
  edm::Wrapper<L1MuGMTReadoutCollection> dummy5;

  edm::Ref<std::vector<L1MuGMTCand> > dummy6;

  edm::RefProd<L1MuGMTReadoutCollection> dummy7;
};
}
