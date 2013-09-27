
#include <vector>
#include "DataFormats/L1TYellow/interface/L1TYellowDigi.h"
#include "DataFormats/L1TYellow/interface/L1TYellowOutput.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    L1TYellowDigiCollection dummya;
    edm::Wrapper<L1TYellowDigiCollection> w_dummya;
    L1TYellowOutputCollection dummyb;
    edm::Wrapper<L1TYellowOutputCollection> w_dummyb;
  };
}
