#include <utility>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/SiteLut.h"

using namespace emtf::phase2::data;

// Static
const int SiteLut::kInvalid = -1;

// Member
SiteLut::SiteLut() {
  lut_[{1, 1, 4}] = 0;   // ME1/1a
  lut_[{1, 1, 1}] = 0;   // ME1/1b
  lut_[{1, 1, 2}] = 1;   // ME1/2
  lut_[{1, 1, 3}] = 1;   // ME1/3
  lut_[{1, 2, 1}] = 2;   // ME2/1
  lut_[{1, 2, 2}] = 2;   // ME2/2
  lut_[{1, 3, 1}] = 3;   // ME3/1
  lut_[{1, 3, 2}] = 3;   // ME3/2
  lut_[{1, 4, 1}] = 4;   // ME4/1
  lut_[{1, 4, 2}] = 4;   // ME4/2
  lut_[{2, 1, 2}] = 5;   // RE1/2
  lut_[{2, 1, 3}] = 5;   // RE1/3
  lut_[{2, 2, 2}] = 6;   // RE2/2
  lut_[{2, 2, 3}] = 6;   // RE2/3
  lut_[{2, 3, 1}] = 7;   // RE3/1
  lut_[{2, 3, 2}] = 7;   // RE3/2
  lut_[{2, 3, 3}] = 7;   // RE3/3
  lut_[{2, 4, 1}] = 8;   // RE4/1
  lut_[{2, 4, 2}] = 8;   // RE4/2
  lut_[{2, 4, 3}] = 8;   // RE4/3
  lut_[{3, 1, 1}] = 9;   // GE1/1
  lut_[{3, 2, 1}] = 10;  // GE2/1
  lut_[{4, 1, 4}] = 11;  // ME0
}

SiteLut::~SiteLut() {
  // Do Nothing
}

void SiteLut::update(const edm::Event&, const edm::EventSetup&) {
  // Do Nothing
}

const int& SiteLut::lookup(const std::tuple<int, int, int>& key) const {
  auto found = lut_.find(key);

  if (found == lut_.end())
    return SiteLut::kInvalid;

  return found->second;
}
