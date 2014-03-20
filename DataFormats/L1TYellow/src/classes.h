
#include <vector>
#include "DataFormats/L1TYellow/interface/YellowDigi.h"
#include "DataFormats/L1TYellow/interface/YellowOutput.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    l1t::YellowDigiCollection dummya;
    edm::Wrapper<l1t::YellowDigiCollection> w_dummya;
    l1t::YellowOutputCollection dummyb;
    edm::Wrapper<l1t::YellowOutputCollection> w_dummyb;
  };
}
