
#include <vector>
#include "DataFormats/L1TYellow/interface/L1TYellowOutput.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    L1TYellowOutputCollection dummy;
    edm::Wrapper<L1TYellowOutputCollection> w_dummy;
  };
}
