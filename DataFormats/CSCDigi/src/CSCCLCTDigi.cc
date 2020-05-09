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
}

uint16_t CSCCLCTDigi::keyStrip(const uint16_t n) const {
  // 10-bit case for strip data word
  if (compCode_ != -1 and n == 8) {
    return keyStrip(4) * 2 + eightStrip();
  }
  // 9-bit case for strip data word
  else if (compCode_ != -1 and n == 4) {
    return keyStrip(2) * 2 + quartStrip();
  }
  // 8-bit case for strip data word (all other cases)
  else {
    return cfeb_ * 32 + strip();
  }
}

uint16_t CSCCLCTDigi::strip() const { return strip_ & kHalfStripMask; }

bool CSCCLCTDigi::quartStrip() const { return (strip_ >> kQuartStripShift) & kQuartStripMask; }

bool CSCCLCTDigi::eightStrip() const { return (strip_ >> kEightStripShift) & kEightStripMask; }

void CSCCLCTDigi::setQuartStrip(const bool quartStrip) {
  // clear the old value
  strip_ &= ~(kQuartStripMask << kQuartStripShift);

  // set the new value
  strip_ |= quartStrip << kQuartStripShift;
}

void CSCCLCTDigi::setEightStrip(const bool eightStrip) {
  // clear the old value
  strip_ &= ~(kEightStripMask << kEightStripShift);

  // set the new value
  strip_ |= eightStrip << kEightStripShift;
}

void CSCCLCTDigi::setRun3(const bool isRun3) { version_ = isRun3 ? Version::Run3 : Version::Legacy; }

bool CSCCLCTDigi::operator>(const CSCCLCTDigi& rhs) const {
  // Several versions of CLCT sorting criteria were used before 2008.
  // They are available in CMSSW versions prior to 3_1_0; here we only keep
  // the latest one, used in TMB-07 firmware (w/o distrips).
  bool returnValue = false;

  uint16_t quality1 = quality();
  uint16_t quality2 = rhs.quality();

  // Run-3 case
  if (version_ == Version::Run3) {
    // Better-quality CLCTs are preferred.
    // If two qualities are equal, smaller bending is preferred;
    // left- and right-bend patterns are considered to be of
    // the same quality. This corresponds to "pattern" being smaller!!!
    // If both qualities and pattern id's are the same, lower keystrip
    // is preferred.
    if ((quality1 > quality2) || (quality1 == quality2 && pattern() < rhs.pattern()) ||
        (quality1 == quality2 && pattern() == rhs.pattern() && keyStrip() < rhs.keyStrip())) {
      returnValue = true;
    }
  }
  // Legacy case:
  else {
    // The bend-direction bit pid[0] is ignored (left and right bends have
    // equal quality).
    uint16_t pattern1 = pattern() & 14;
    uint16_t pattern2 = rhs.pattern() & 14;

    // Better-quality CLCTs are preferred.
    // If two qualities are equal, larger pattern id (i.e., straighter pattern)
    // is preferred; left- and right-bend patterns are considered to be of
    // the same quality.
    // If both qualities and pattern id's are the same, lower keystrip
    // is preferred.
    if ((quality1 > quality2) || (quality1 == quality2 && pattern1 > pattern2) ||
        (quality1 == quality2 && pattern1 == pattern2 && keyStrip() < rhs.keyStrip())) {
      returnValue = true;
    }
  }
  return returnValue;
}

bool CSCCLCTDigi::operator==(const CSCCLCTDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && quality() == rhs.quality() && pattern() == rhs.pattern() &&
      keyStrip() == rhs.keyStrip() && stripType() == rhs.stripType() && bend() == bend() && bx() == rhs.bx() &&
      compCode() == rhs.compCode()) {
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
    char stripTypeChar = (stripType() == 0) ? 'D' : 'H';
    char bendChar = (bend() == 0) ? 'L' : 'R';

    edm::LogVerbatim("CSCDigi") << " CSC CLCT #" << std::setw(1) << trackNumber() << ": Valid = " << std::setw(1)
                                << isValid() << " Key Strip = " << std::setw(3) << keyStrip()
                                << " Strip = " << std::setw(2) << strip() << " Quality = " << std::setw(1) << quality()
                                << " Pattern = " << std::setw(1) << pattern() << " Bend = " << std::setw(1) << bendChar
                                << " Strip type = " << std::setw(1) << stripTypeChar << " CFEB ID = " << std::setw(1)
                                << cfeb() << " BX = " << std::setw(1) << bx() << " Full BX= " << std::setw(1)
                                << fullBX() << " Comp Code= " << std::setw(1) << compCode();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid Cathode LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCCLCTDigi& digi) {
  return o << "CSC CLCT #" << digi.trackNumber() << ": Valid = " << digi.isValid() << " Quality = " << digi.quality()
           << " Pattern = " << digi.pattern() << " StripType = " << digi.stripType() << " Bend = " << digi.bend()
           << " Strip = " << digi.strip() << " KeyStrip = " << digi.keyStrip() << " CFEB = " << digi.cfeb()
           << " BX = " << digi.bx() << " Comp Code " << digi.compCode();
}
