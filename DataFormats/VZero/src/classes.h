#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/VZero/interface/VZero.h"
#include <vector>

namespace {
  struct dictionary {
    reco::VZeroCollection v9;
    edm::Wrapper<reco::VZeroCollection> c9;
  };
}
