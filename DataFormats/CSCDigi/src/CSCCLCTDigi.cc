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
CSCCLCTDigi::CSCCLCTDigi(const int valid,
                         const int quality,
                         const int pattern,
                         const int striptype,
                         const int bend,
                         const int strip,
                         const int cfeb,
                         const int bx,
                         const int trknmb,
                         const int fullbx,
                         const int compCode)
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
      compCode_(compCode) {
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
      compCode_(-1) {
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

int CSCCLCTDigi::getKeyStrip(int n) const {
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

int CSCCLCTDigi::getStrip() const { return strip_ & kHalfStripMask; }

bool CSCCLCTDigi::getQuartStrip() const { return (strip_ >> kQuartStripShift) & kQuartStripMask; }

bool CSCCLCTDigi::getEightStrip() const { return (strip_ >> kEightStripShift) & kEightStripMask; }

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

bool CSCCLCTDigi::operator>(const CSCCLCTDigi& rhs) const {
  // Several versions of CLCT sorting criteria were used before 2008.
  // They are available in CMSSW versions prior to 3_1_0; here we only keep
  // the latest one, used in TMB-07 firmware (w/o distrips).
  bool returnValue = false;

  int quality1 = getQuality();
  int quality2 = rhs.getQuality();
  // The bend-direction bit pid[0] is ignored (left and right bends have
  // equal quality).
  int pattern1 = getPattern() & 14;
  int pattern2 = rhs.getPattern() & 14;

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

  return returnValue;
}

bool CSCCLCTDigi::operator==(const CSCCLCTDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && getQuality() == rhs.getQuality() && getPattern() == rhs.getPattern() &&
      getKeyStrip() == rhs.getKeyStrip() && getStripType() == rhs.getStripType() && getBend() == getBend() &&
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

std::ostream& operator<<(std::ostream& o, const CSCCLCTDigi& digi) {
  return o << "CSC CLCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " Pattern = " << digi.getPattern() << " StripType = " << digi.getStripType()
           << " Bend = " << digi.getBend() << " Strip = " << digi.getStrip() << " KeyStrip = " << digi.getKeyStrip()
           << " CFEB = " << digi.getCFEB() << " BX = " << digi.getBX() << " Comp Code " << digi.getCompCode();
}
