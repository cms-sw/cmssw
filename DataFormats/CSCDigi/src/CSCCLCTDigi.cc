/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives.
 *
 *
 * \author N. Terentiev, CMU
 */

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

enum Pattern_Info { NUM_LAYERS = 6, CLCT_PATTERN_WIDTH = 11 };

/// Constructors
CSCCLCTDigi::CSCCLCTDigi(const uint16_t valid,
                         const uint16_t quality,
                         const uint16_t pattern,
                         const uint16_t striptype,
                         const uint16_t bend,
                         const uint16_t strip,
                         const uint16_t cfeb,
                         const uint16_t bx,
                         const uint16_t trknmb,
                         const uint16_t fullbx,
                         const int16_t compCode,
                         const Version version)
    : valid_(valid),
      quality_(quality),
      pattern_(pattern),
      striptype_(striptype),
      bend_(bend),
      strip_(strip),
      cfeb_(cfeb),
      bx_(bx),
      trknmb_(trknmb),
      fullbx_(fullbx),
      compCode_(compCode),
      version_(version) {
  hits_.resize(NUM_LAYERS);
  for (auto& p : hits_) {
    p.resize(CLCT_PATTERN_WIDTH);
  }
}

/// Default
CSCCLCTDigi::CSCCLCTDigi()
    : valid_(0),
      quality_(0),
      pattern_(0),
      striptype_(0),
      bend_(0),
      strip_(0),
      cfeb_(0),
      bx_(0),
      trknmb_(0),
      fullbx_(0),
      compCode_(-1),
      version_(Version::Legacy) {
  hits_.resize(NUM_LAYERS);
  for (auto& p : hits_) {
    p.resize(CLCT_PATTERN_WIDTH);
  }
}

/// Clears this CLCT.
void CSCCLCTDigi::clear() {
  valid_ = 0;
  quality_ = 0;
  pattern_ = 0;
  striptype_ = 0;
  bend_ = 0;
  strip_ = 0;
  cfeb_ = 0;
  bx_ = 0;
  trknmb_ = 0;
  fullbx_ = 0;
  compCode_ = -1;
  hits_.clear();
  hits_.resize(NUM_LAYERS);
  for (auto& p : hits_) {
    p.resize(CLCT_PATTERN_WIDTH);
  }
  setSlope(0);
}

uint16_t CSCCLCTDigi::getPattern() const { return getDataWord(pattern_, kLegacyPatternShift, kLegacyPatternMask); }

void CSCCLCTDigi::setPattern(const uint16_t pattern) {
  setDataWord(pattern, pattern_, kLegacyPatternShift, kLegacyPatternMask);
}

uint16_t CSCCLCTDigi::getRun3Pattern() const {
  if (!isRun3())
    return 0;
  return getDataWord(pattern_, kRun3PatternShift, kRun3PatternMask);
}

void CSCCLCTDigi::setRun3Pattern(const uint16_t pattern) {
  if (!isRun3())
    return;
  setDataWord(pattern, pattern_, kRun3PatternShift, kRun3PatternMask);
}

uint16_t CSCCLCTDigi::getSlope() const {
  if (!isRun3())
    return 0;
  return getDataWord(pattern_, kRun3SlopeShift, kRun3SlopeMask);
}

void CSCCLCTDigi::setSlope(const uint16_t slope) {
  if (!isRun3())
    return;
  setDataWord(slope, pattern_, kRun3SlopeShift, kRun3SlopeMask);
}

// slope in number of half-strips/layer
float CSCCLCTDigi::getFractionalSlope(const uint16_t nBits) const {
  if (isRun3()) {
    const float minSlope = 0;
    const float maxSlope = 2.5;
    const int range = pow(2, nBits);
    const float deltaSlope = (maxSlope - minSlope) / range;
    const float slopeValue = minSlope + deltaSlope * getSlope();
    return (2 * getBend() - 1) * slopeValue;
  } else {
    int slope[11] = {0, 0, -8, 8, -6, 6, -4, 4, -2, 2, 0};
    return float(slope[getPattern()] / 5.);
  }
}

uint16_t CSCCLCTDigi::getKeyStrip(const uint16_t n) const {
  // 10-bit case for strip data word
  if (compCode_ != -1 and n == 8) {
    return getKeyStrip(4) * 2 + getEightStrip();
  }
  // 9-bit case for strip data word
  else if (compCode_ != -1 and n == 4) {
    return getKeyStrip(2) * 2 + getQuartStrip();
  }
  // 8-bit case for strip data word (all other cases)
  else {
    return cfeb_ * 32 + getStrip();
  }
}

/// return the fractional strip (middle of the strip)
float CSCCLCTDigi::getFractionalStrip(const uint16_t n) const {
  if (compCode_ != -1 and n == 8) {
    return 0.125f * (getKeyStrip(n) + 0.5);
  } else if (compCode_ != -1 and n == 4) {
    return 0.25f * (getKeyStrip(n) + 0.5);
  } else {
    return 0.5f * (getKeyStrip(n) + 0.5);
  }
}

uint16_t CSCCLCTDigi::getStrip() const { return getDataWord(strip_, kHalfStripShift, kHalfStripMask); }

bool CSCCLCTDigi::getQuartStrip() const {
  if (!isRun3())
    return false;
  return getDataWord(strip_, kQuartStripShift, kQuartStripMask);
}

bool CSCCLCTDigi::getEightStrip() const {
  if (!isRun3())
    return false;
  return getDataWord(strip_, kEightStripShift, kEightStripMask);
}

void CSCCLCTDigi::setQuartStrip(const bool quartStrip) {
  if (!isRun3())
    return;
  setDataWord(quartStrip, strip_, kQuartStripShift, kQuartStripMask);
}

void CSCCLCTDigi::setEightStrip(const bool eightStrip) {
  if (!isRun3())
    return;
  setDataWord(eightStrip, strip_, kEightStripShift, kEightStripMask);
}

void CSCCLCTDigi::setRun3(const bool isRun3) { version_ = isRun3 ? Version::Run3 : Version::Legacy; }

bool CSCCLCTDigi::operator>(const CSCCLCTDigi& rhs) const {
  // Several versions of CLCT sorting criteria were used before 2008.
  // They are available in CMSSW versions prior to 3_1_0; here we only keep
  // the latest one, used in TMB-07 firmware (w/o distrips).
  bool returnValue = false;

  uint16_t quality1 = getQuality();
  uint16_t quality2 = rhs.getQuality();

  // Run-3 case
  if (version_ == Version::Run3) {
    // Better-quality CLCTs are preferred.
    // If two qualities are equal, smaller bending is preferred;
    // left- and right-bend patterns are considered to be of
    // the same quality. This corresponds to "pattern" being smaller!!!
    // If both qualities and pattern id's are the same, lower keystrip
    // is preferred.
    if ((quality1 > quality2) || (quality1 == quality2 && getPattern() < rhs.getPattern()) ||
        (quality1 == quality2 && getPattern() == rhs.getPattern() && getKeyStrip() < rhs.getKeyStrip())) {
      returnValue = true;
    }
  }
  // Legacy case:
  else {
    // The bend-direction bit pid[0] is ignored (left and right bends have
    // equal quality).
    uint16_t pattern1 = getPattern() & 14;
    uint16_t pattern2 = rhs.getPattern() & 14;

    // Better-quality CLCTs are preferred.
    // If two qualities are equal, larger pattern id (i.e., straighter pattern)
    // is preferred; left- and right-bend patterns are considered to be of
    // the same quality.
    // If both qualities and pattern id's are the same, lower keystrip
    // is preferred.
    if ((quality1 > quality2) || (quality1 == quality2 && pattern1 > pattern2) ||
        (quality1 == quality2 && pattern1 == pattern2 && getKeyStrip() < rhs.getKeyStrip())) {
      returnValue = true;
    }
  }
  return returnValue;
}

bool CSCCLCTDigi::operator==(const CSCCLCTDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && getQuality() == rhs.getQuality() && getPattern() == rhs.getPattern() &&
      getKeyStrip() == rhs.getKeyStrip() && getStripType() == rhs.getStripType() && getBend() == rhs.getBend() &&
      getBX() == rhs.getBX() && getCompCode() == rhs.getCompCode()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCCLCTDigi::operator!=(const CSCCLCTDigi& rhs) const {
  // True if == is false.
  bool returnValue = true;
  if ((*this) == rhs)
    returnValue = false;
  return returnValue;
}

/// Debug
void CSCCLCTDigi::print() const {
  if (isValid()) {
    char stripType = (getStripType() == 0) ? 'D' : 'H';
    char bend = (getBend() == 0) ? 'L' : 'R';

    edm::LogVerbatim("CSCDigi") << " CSC CLCT #" << std::setw(1) << getTrknmb() << ": Valid = " << std::setw(1)
                                << isValid() << " Key Strip = " << std::setw(3) << getKeyStrip()
                                << " Strip = " << std::setw(2) << getStrip() << " Quality = " << std::setw(1)
                                << getQuality() << " Pattern = " << std::setw(1) << getPattern()
                                << " Bend = " << std::setw(1) << bend << " Strip type = " << std::setw(1) << stripType
                                << " CFEB ID = " << std::setw(1) << getCFEB() << " BX = " << std::setw(1) << getBX()
                                << " Full BX= " << std::setw(1) << getFullBX() << " Comp Code= " << std::setw(1)
                                << getCompCode();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid Cathode LCT.";
  }
}

void CSCCLCTDigi::setDataWord(const uint16_t newWord, uint16_t& word, const unsigned shift, const unsigned mask) {
  // clear the old value
  word &= ~(mask << shift);

  // set the new value
  word |= newWord << shift;
}

uint16_t CSCCLCTDigi::getDataWord(const uint16_t word, const unsigned shift, const unsigned mask) const {
  return (word >> shift) & mask;
}

std::ostream& operator<<(std::ostream& o, const CSCCLCTDigi& digi) {
  return o << "CSC CLCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " Pattern = " << digi.getPattern() << " StripType = " << digi.getStripType()
           << " Bend = " << digi.getBend() << " Strip = " << digi.getStrip() << " KeyStrip = " << digi.getKeyStrip()
           << " CFEB = " << digi.getCFEB() << " BX = " << digi.getBX() << " Comp Code " << digi.getCompCode();
}
