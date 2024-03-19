
#include <utility>

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/ZoneLut.h"

using namespace emtf::phase2::data;

ZoneLut::ZoneLut() {
  auto& zone0 = zones_.emplace_back();
  zone0.lut_[0] = {4, 26};   // ME1/1
  zone0.lut_[3] = {4, 25};   // ME2/1
  zone0.lut_[5] = {4, 25};   // ME3/1
  zone0.lut_[7] = {4, 25};   // ME4/1
  zone0.lut_[9] = {17, 26};  // GE1/1
  zone0.lut_[12] = {7, 25};  // GE2/1
  zone0.lut_[14] = {4, 25};  // RE3/1
  zone0.lut_[16] = {4, 25};  // RE4/1
  zone0.lut_[18] = {4, 23};  // ME0

  auto& zone1 = zones_.emplace_back();
  zone1.lut_[0] = {24, 53};   // ME1/1
  zone1.lut_[1] = {46, 54};   // ME1/2
  zone1.lut_[3] = {23, 49};   // ME2/1
  zone1.lut_[5] = {23, 41};   // ME3/1
  zone1.lut_[6] = {44, 54};   // ME3/2
  zone1.lut_[7] = {23, 35};   // ME4/1
  zone1.lut_[8] = {38, 54};   // ME4/2
  zone1.lut_[9] = {24, 52};   // GE1/1
  zone1.lut_[10] = {52, 56};  // RE1/2
  zone1.lut_[12] = {23, 46};  // GE2/1
  zone1.lut_[14] = {23, 36};  // RE3/1
  zone1.lut_[15] = {40, 52};  // RE3/2
  zone1.lut_[16] = {23, 31};  // RE4/1
  zone1.lut_[17] = {35, 54};  // RE4/2

  auto& zone2 = zones_.emplace_back();
  zone2.lut_[1] = {52, 88};   // ME1/2
  zone2.lut_[4] = {52, 88};   // ME2/2
  zone2.lut_[6] = {50, 88};   // ME3/2
  zone2.lut_[8] = {50, 88};   // ME4/2
  zone2.lut_[10] = {52, 84};  // RE1/2
  zone2.lut_[13] = {52, 88};  // RE2/2
  zone2.lut_[15] = {48, 84};  // RE3/2
  zone2.lut_[17] = {52, 84};  // RE4/2
}

ZoneLut::~ZoneLut() {
  // Do Nothing
}

void ZoneLut::update(const edm::Event&, const edm::EventSetup&) {
  // Do Nothing
}

int ZoneLut::getZones(const int& host, const int& theta) const {
  int i = 0;
  int word = 0;

  for (const auto& zone : zones_) {
    bool in_zone = zone.contains(host, theta);

    if (in_zone) {
      word |= (1u << i);
    }

    ++i;
  }

  return word;
}

int ZoneLut::getZones(const int& host, const int& theta1, const int& theta2) const {
  int i = 0;
  int word = 0;

  for (const auto& zone : zones_) {
    bool in_zone = zone.contains(host, theta1, theta2);

    if (in_zone) {
      word |= (1u << i);
    }

    ++i;
  }

  return word;
}

bool Zone::contains(const int& host, const int& theta) const {
  // Short-Circuit: LUT not found
  auto found = lut_.find(host);

  if (found == lut_.end())
    return false;

  // Short-Circuit: Must be within theta range
  auto& theta_range = found->second;

  if (theta_range.first <= theta && theta <= theta_range.second) {
    return true;
  }

  return false;
}

bool Zone::contains(const int& host, const int& theta1, const int& theta2) const {
  // Short-Circuit: LUT not found
  auto found = lut_.find(host);

  if (found == lut_.end())
    return false;

  // Short-Circuit: Must be within theta range
  auto& theta_range = found->second;

  if (theta_range.first <= theta1 && theta1 <= theta_range.second) {
    return true;
  }

  if (theta_range.first <= theta2 && theta2 <= theta_range.second) {
    return true;
  }

  return false;
}
