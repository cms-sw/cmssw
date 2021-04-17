/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives.
 *
 *
 * \author L.Gray, UF
 */

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

/// Constructors
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(const uint16_t itrknmb,
                                           const uint16_t ivalid,
                                           const uint16_t iquality,
                                           const uint16_t ikeywire,
                                           const uint16_t istrip,
                                           const uint16_t ipattern,
                                           const uint16_t ibend,
                                           const uint16_t ibx,
                                           const uint16_t impclink,
                                           const uint16_t ibx0,
                                           const uint16_t isyncErr,
                                           const uint16_t icscID,
                                           const uint16_t ihmt,
                                           const Version version)
    : trknmb(itrknmb),
      valid(ivalid),
      quality(iquality),
      keywire(ikeywire),
      strip(istrip),
      pattern(ipattern),
      bend(ibend),
      bx(ibx),
      mpclink(impclink),
      bx0(ibx0),
      syncErr(isyncErr),
      cscID(icscID),
      hmt(ihmt),
      version_(version) {}

/// Default
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi() {
  clear();  // set contents to zero
}

/// Clears this LCT.
void CSCCorrelatedLCTDigi::clear() {
  trknmb = 0;
  valid = 0;
  quality = 0;
  keywire = 0;
  strip = 0;
  pattern = 0;
  bend = 0;
  bx = 0;
  mpclink = 0;
  bx0 = 0;
  syncErr = 0;
  cscID = 0;
  hmt = 0;
  version_ = Version::Legacy;
  // clear the components
  type_ = 0;
  alct_.clear();
  clct_.clear();
  gem1_ = GEMPadDigi();
  gem2_ = GEMPadDigi();
}

uint16_t CSCCorrelatedLCTDigi::getStrip(const uint16_t n) const {
  // all 10 bits
  if (n == 8) {
    return 2 * getStrip(4) + getEighthStrip();
  }
  // lowest 9 bits
  else if (n == 4) {
    return 2 * getStrip(2) + getQuartStrip();
  }
  // lowest 8 bits
  else {
    return strip & kHalfStripMask;
  }
}

void CSCCorrelatedLCTDigi::setQuartStrip(const bool quartStrip) {
  if (!isRun3())
    return;
  setDataWord(quartStrip, strip, kQuartStripShift, kQuartStripMask);
}

void CSCCorrelatedLCTDigi::setEighthStrip(const bool eighthStrip) {
  if (!isRun3())
    return;
  setDataWord(eighthStrip, strip, kEighthStripShift, kEighthStripMask);
}

bool CSCCorrelatedLCTDigi::getQuartStrip() const {
  if (!isRun3())
    return false;
  return getDataWord(strip, kQuartStripShift, kQuartStripMask);
}

bool CSCCorrelatedLCTDigi::getEighthStrip() const {
  if (!isRun3())
    return false;
  return getDataWord(strip, kEighthStripShift, kEighthStripMask);
}

uint16_t CSCCorrelatedLCTDigi::getSlope() const {
  if (!isRun3())
    return 0;
  return getDataWord(pattern, kRun3SlopeShift, kRun3SlopeMask);
}

void CSCCorrelatedLCTDigi::setSlope(const uint16_t slope) {
  if (!isRun3())
    return;
  setDataWord(slope, pattern, kRun3SlopeShift, kRun3SlopeMask);
}

// slope in number of half-strips/layer
float CSCCorrelatedLCTDigi::getFractionalSlope() const {
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

/// return the fractional strip
float CSCCorrelatedLCTDigi::getFractionalStrip(const uint16_t n) const {
  if (n == 8) {
    return 0.125f * (getStrip(n) + 0.5);
  } else if (n == 4) {
    return 0.25f * (getStrip(n) + 0.5);
  } else {
    return 0.5f * (getStrip(n) + 0.5);
  }
}

uint16_t CSCCorrelatedLCTDigi::getCLCTPattern() const {
  return (isRun3() ? std::numeric_limits<uint16_t>::max() : (pattern & 0xF));
}

uint16_t CSCCorrelatedLCTDigi::getPattern() const {
  return getDataWord(pattern, kLegacyPatternShift, kLegacyPatternMask);
}

void CSCCorrelatedLCTDigi::setPattern(const uint16_t pat) {
  setDataWord(pat, pattern, kLegacyPatternShift, kLegacyPatternMask);
}

uint16_t CSCCorrelatedLCTDigi::getRun3Pattern() const {
  if (!isRun3())
    return 0;
  return getDataWord(pattern, kRun3PatternShift, kRun3PatternMask);
}

void CSCCorrelatedLCTDigi::setRun3Pattern(const uint16_t pat) {
  if (!isRun3())
    return;
  setDataWord(pat, pattern, kRun3PatternShift, kRun3PatternMask);
}

uint16_t CSCCorrelatedLCTDigi::getHMT() const { return (isRun3() ? hmt : std::numeric_limits<uint16_t>::max()); }

void CSCCorrelatedLCTDigi::setHMT(const uint16_t h) { hmt = isRun3() ? h : std::numeric_limits<uint16_t>::max(); }

void CSCCorrelatedLCTDigi::setRun3(const bool isRun3) { version_ = isRun3 ? Version::Run3 : Version::Legacy; }

/// Comparison
bool CSCCorrelatedLCTDigi::operator==(const CSCCorrelatedLCTDigi& rhs) const {
  return ((trknmb == rhs.trknmb) && (quality == rhs.quality) && (keywire == rhs.keywire) && (strip == rhs.strip) &&
          (pattern == rhs.pattern) && (bend == rhs.bend) && (bx == rhs.bx) && (valid == rhs.valid) &&
          (mpclink == rhs.mpclink) && (hmt == rhs.hmt));
}

/// Debug
void CSCCorrelatedLCTDigi::print() const {
  if (isValid()) {
    edm::LogVerbatim("CSCDigi") << "CSC LCT #" << getTrknmb() << ": Valid = " << isValid()
                                << " Quality = " << getQuality() << " Key Wire = " << getKeyWG()
                                << " Strip = " << getStrip() << " Pattern = " << getPattern()
                                << " Bend = " << ((getBend() == 0) ? 'L' : 'R') << " BX = " << getBX()
                                << " MPC Link = " << getMPCLink() << " Type (SIM) = " << getType()
                                << " HMT Bit = " << getHMT();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid correlated LCT.";
  }
}

void CSCCorrelatedLCTDigi::setDataWord(const uint16_t newWord,
                                       uint16_t& word,
                                       const unsigned shift,
                                       const unsigned mask) {
  // clear the old value
  word &= ~(mask << shift);

  // set the new value
  word |= newWord << shift;
}

uint16_t CSCCorrelatedLCTDigi::getDataWord(const uint16_t word, const unsigned shift, const unsigned mask) const {
  return (word >> shift) & mask;
}

std::ostream& operator<<(std::ostream& o, const CSCCorrelatedLCTDigi& digi) {
  return o << "CSC LCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " MPC Link = " << digi.getMPCLink() << " cscID = " << digi.getCSCID()
           << " syncErr = " << digi.getSyncErr() << " Type (SIM) = " << digi.getType() << " HMT Bit = " << digi.getHMT()
           << "\n"
           << "  cathode info: Strip = " << digi.getStrip() << " Pattern = " << digi.getPattern()
           << " Bend = " << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           << "    anode info: Key wire = " << digi.getKeyWG() << " BX = " << digi.getBX()
           << " bx0 = " << digi.getBX0();
}
