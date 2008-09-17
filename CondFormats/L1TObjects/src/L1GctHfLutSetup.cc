
#include "CondFormats/L1TObjects/interface/L1GctHfLutSetup.h"

L1GctHfLutSetup::L1GctHfLutSetup() :
  m_thresholds()
{
}

L1GctHfLutSetup::~L1GctHfLutSetup() {}

void L1GctHfLutSetup::setThresholds(const hfLutType type, const std::vector<unsigned> thr)
{
  // Set thresholds for a particular Lut type.
  // Thresholds should (of course) be ordered - but no check is performed.
  // The number of thresholds is one fewer than the number of possible
  // output codes (e.g. 3 bits -> 7 thresholds)
  m_thresholds[type].resize(kHfOutputMaxValue);
  for (unsigned i=0; i<kHfOutputMaxValue; ++i) {
    if (i<thr.size()) {
      m_thresholds[type].at(i) = static_cast<uint16_t>(thr.at(i));
    } else {
      m_thresholds[type].at(i) = kHfEtSumMaxValue;
    }
  }
}

uint16_t L1GctHfLutSetup::outputValue(const hfLutType type, const uint16_t inputValue) const
{
  // Calculate Lut contents by comparison against a set of thresholds.
  // Check that the Lut type requested has actually been setup - otherwise
  // we return the max possible output value
  uint16_t result = kHfOutputMaxValue;
  std::map<hfLutType, std::vector<uint16_t> >::const_iterator thr = m_thresholds.find(type);
  if (thr != m_thresholds.end()) {
    for (unsigned i=0; i<kHfOutputMaxValue; ++i) {
      if (inputValue < (thr->second).at(i)) {
	result = i;
	break;
      }
    }
  }
  return result;
}
