#include <utility>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/HostLut.h"

using namespace emtf::phase2::data;

// Static
const int HostLut::kInvalid = -1;

// Member
HostLut::HostLut() {
  lut_[{1, 1, 4}] = 0;   // ME1/1a
  lut_[{1, 1, 1}] = 0;   // ME1/1b
  lut_[{1, 1, 2}] = 1;   // ME1/2
  lut_[{1, 1, 3}] = 2;   // ME1/3
  lut_[{1, 2, 1}] = 3;   // ME2/1
  lut_[{1, 2, 2}] = 4;   // ME2/2
  lut_[{1, 3, 1}] = 5;   // ME3/1
  lut_[{1, 3, 2}] = 6;   // ME3/2
  lut_[{1, 4, 1}] = 7;   // ME4/1
  lut_[{1, 4, 2}] = 8;   // ME4/2
  lut_[{3, 1, 1}] = 9;   // GE1/1
  lut_[{2, 1, 2}] = 10;  // RE1/2
  lut_[{2, 1, 3}] = 11;  // RE1/3
  lut_[{3, 2, 1}] = 12;  // GE2/1
  lut_[{2, 2, 2}] = 13;  // RE2/2
  lut_[{2, 2, 3}] = 13;  // RE2/3
  lut_[{2, 3, 1}] = 14;  // RE3/1
  lut_[{2, 3, 2}] = 15;  // RE3/2
  lut_[{2, 3, 3}] = 15;  // RE3/3
  lut_[{2, 4, 1}] = 16;  // RE4/1
  lut_[{2, 4, 2}] = 17;  // RE4/2
  lut_[{2, 4, 3}] = 17;  // RE4/3
  lut_[{4, 1, 4}] = 18;  // ME0
}

HostLut::~HostLut() {
  // Do Nothing
}

void HostLut::update(const edm::Event&, const edm::EventSetup&) {
  // Do Nothing
}

const int& HostLut::lookup(const std::tuple<int, int, int>& key) const {
  auto found = lut_.find(key);

  if (found == lut_.end())
    return HostLut::kInvalid;

  return found->second;
}
