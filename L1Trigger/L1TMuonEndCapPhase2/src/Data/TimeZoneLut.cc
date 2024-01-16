#include <utility>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/TimeZoneLut.h"

using namespace emtf::phase2::data;

// Static
bool TimeZoneLut::in_range(const std::pair<int, int>& range, const int& bx) const {
  return range.first <= bx && bx <= range.second;
}

// Member
TimeZoneLut::TimeZoneLut() {
  lut_[0] = {-1, 0};   // ME1/1
  lut_[1] = {-1, 0};   // ME1/2
  lut_[2] = {-1, 0};   // ME1/3
  lut_[3] = {-1, 0};   // ME2/1
  lut_[4] = {-1, 0};   // ME2/2
  lut_[5] = {-1, 0};   // ME3/1
  lut_[6] = {-1, 0};   // ME3/2
  lut_[7] = {-1, 0};   // ME4/1
  lut_[8] = {-1, 0};   // ME4/2
  lut_[9] = {-1, 0};   // GE1/1
  lut_[10] = {0, 0};   // RE1/2
  lut_[11] = {0, 0};   // RE1/3
  lut_[12] = {-1, 0};  // GE2/1
  lut_[13] = {0, 0};   // RE2/2
  lut_[14] = {0, 0};   // RE3/1
  lut_[15] = {0, 0};   // RE3/2
  lut_[16] = {0, 0};   // RE4/1
  lut_[17] = {0, 0};   // RE4/2
  lut_[18] = {0, 0};   // ME0
}

TimeZoneLut::~TimeZoneLut() {
  // Do Nothing
}

void TimeZoneLut::update(const edm::Event&, const edm::EventSetup&) {
  // Do Nothing
}

int TimeZoneLut::get_timezones(const int& host, const int& bx) const {
  auto found = lut_.find(host);

  // Short-Circuit: Host doesn't exist
  if (found == lut_.end())
    return 0x0;

  // Build word
  int word = 0x0;

  word |= in_range(found->second, bx) ? 0b001 : 0;
  word |= in_range(found->second, bx + 1) ? 0b010 : 0;  // +1 BX delay
  word |= in_range(found->second, bx + 2) ? 0b100 : 0;  // +2 BX delay

  return word;
}
