
#include <vector>
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    L1CaloEmCollection em;
    L1CaloRegionCollection rgn;

    edm::Wrapper<L1CaloEmCollection> w_em;
    edm::Wrapper<L1CaloRegionCollection> w_rgn;
  };
}
