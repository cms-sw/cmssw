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
                         const Version version,
                         const bool run3_quart_strip_bit,
                         const bool run3_eighth_strip_bit,
                         const uint16_t run3_pattern,
                         const uint16_t run3_slope)
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
      run3_quart_strip_bit_(run3_quart_strip_bit),
      run3_eighth_strip_bit_(run3_eighth_strip_bit),
      run3_pattern_(run3_pattern),
      run3_slope_(run3_slope),
      version_(version) {
  hits_.resize(NUM_LAYERS);
  for (auto& p : hits_) {
    p.resize(CLCT_PATTERN_WIDTH);
  }
}

/// Default
CSCCLCTDigi::CSCCLCTDigi() {
  clear();  // set contents to zero
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
  // Run-3 variables
  compCode_ = -1;
  run3_quart_strip_bit_ = false;
  run3_eighth_strip_bit_ = false;
  run3_pattern_ = 0;
  run3_slope_ = 0;
  version_ = Version::Legacy;
  hits_.clear();
  hits_.resize(NUM_LAYERS);
  for (auto& p : hits_) {
    p.resize(CLCT_PATTERN_WIDTH);
  }
}

// slope in number of half-strips/layer
float CSCCLCTDigi::getFractionalSlope() const {
  if (isRun3()) {
    // 4-bit slope
    float slope[17] = {
        0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 2.0, 2.5};
    return (2 * getBend() - 1) * slope[getSlope()];
  } else {
    int slope[11] = {0, 0, -8, 8, -6, 6, -4, 4, -2, 2, 0};
    return float(slope[getPattern()] / 5.);
  }
}

uint16_t CSCCLCTDigi::getKeyStrip(const uint16_t n) const {
  // 10-bit case for strip data word
  if (compCode_ != -1 and n == 8) {
    return getKeyStrip(4) * 2 + getEighthStripBit();
  }
  // 9-bit case for strip data word
  else if (compCode_ != -1 and n == 4) {
    return getKeyStrip(2) * 2 + getQuartStripBit();
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

std::ostream& operator<<(std::ostream& o, const CSCCLCTDigi& digi) {
  if (digi.isRun3())
    return o << "CSC CLCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " BX = " << digi.getBX()
             << " Run-2 Pattern = " << digi.getPattern() << " Run-3 Pattern = " << digi.getRun3Pattern()
             << " Quality = " << digi.getQuality() << " Comp Code " << digi.getCompCode()
             << " Bend = " << digi.getBend() << "\n"
             << " Slope = " << digi.getSlope() << " CFEB = " << digi.getCFEB() << " Strip = " << digi.getStrip()
             << " KeyHalfStrip = " << digi.getKeyStrip() << " KeyQuartStrip = " << digi.getKeyStrip(4)
             << " KeyEighthStrip = " << digi.getKeyStrip(8);
  else
    return o << "CSC CLCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " BX = " << digi.getBX()
             << " Pattern = " << digi.getPattern() << " Quality = " << digi.getQuality() << " Bend = " << digi.getBend()
             << " CFEB = " << digi.getCFEB() << " HalfStrip = " << digi.getStrip()
             << " KeyHalfStrip = " << digi.getKeyStrip();
}
