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
                                           const Version version,
                                           const bool run3_quart_strip_bit,
                                           const bool run3_eighth_strip_bit,
                                           const uint16_t run3_pattern,
                                           const uint16_t run3_slope,
                                           const int type)
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
      hmt(0),
      run3_quart_strip_bit_(run3_quart_strip_bit),
      run3_eighth_strip_bit_(run3_eighth_strip_bit),
      run3_pattern_(run3_pattern),
      run3_slope_(run3_slope),
      type_(type),
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
  // Run-3 variables
  run3_quart_strip_bit_ = false;
  run3_eighth_strip_bit_ = false;
  run3_pattern_ = 0;
  run3_slope_ = 0;
  // clear the components
  type_ = 1;
  alct_.clear();
  clct_.clear();
  gem1_ = GEMPadDigi();
  gem2_ = GEMPadDigi();
}

uint16_t CSCCorrelatedLCTDigi::getStrip(const uint16_t n) const {
  // all 10 bits
  if (n == 8) {
    return 2 * getStrip(4) + getEighthStripBit();
  }
  // lowest 9 bits
  else if (n == 4) {
    return 2 * getStrip(2) + getQuartStripBit();
  }
  // lowest 8 bits
  else {
    return strip;
  }
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

uint16_t CSCCorrelatedLCTDigi::getHMT() const { return (isRun3() ? hmt : std::numeric_limits<uint16_t>::max()); }

void CSCCorrelatedLCTDigi::setHMT(const uint16_t h) { hmt = isRun3() ? h : std::numeric_limits<uint16_t>::max(); }

void CSCCorrelatedLCTDigi::setRun3(const bool isRun3) { version_ = isRun3 ? Version::Run3 : Version::Legacy; }

/// Comparison
bool CSCCorrelatedLCTDigi::operator==(const CSCCorrelatedLCTDigi& rhs) const {
  return ((trknmb == rhs.trknmb) && (quality == rhs.quality) && (keywire == rhs.keywire) && (strip == rhs.strip) &&
          (pattern == rhs.pattern) && (bend == rhs.bend) && (bx == rhs.bx) && (valid == rhs.valid) &&
          (mpclink == rhs.mpclink));
}

/// Debug
void CSCCorrelatedLCTDigi::print() const {
  if (isValid()) {
    edm::LogVerbatim("CSCDigi") << "CSC LCT #" << getTrknmb() << ": Valid = " << isValid()
                                << " Quality = " << getQuality() << " Key Wire = " << getKeyWG()
                                << " Strip = " << getStrip() << " Pattern = " << getPattern()
                                << " Bend = " << ((getBend() == 0) ? 'L' : 'R') << " BX = " << getBX()
                                << " MPC Link = " << getMPCLink() << " Type (SIM) = " << getType();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid correlated LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCCorrelatedLCTDigi& digi) {
  // do not print out CSCID and sync error. They are not used anyway in the firmware, or the emulation
  if (digi.isRun3())
    return o << "CSC LCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " BX = " << digi.getBX()
             << " Run-2 Pattern = " << digi.getPattern() << " Run-3 Pattern = " << digi.getRun3Pattern()
             << " Quality = " << digi.getQuality() << " Bend = " << digi.getBend() << " Slope = " << digi.getSlope()
             << "\n"
             << " KeyHalfStrip = " << digi.getStrip() << " KeyQuartStrip = " << digi.getStrip(4)
             << " KeyEighthStrip = " << digi.getStrip(8) << " KeyWireGroup = " << digi.getKeyWG()
             << " Type (SIM) = " << digi.getType() << " MPC Link = " << digi.getMPCLink();
  else
    return o << "CSC LCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " BX = " << digi.getBX()
             << " Pattern = " << digi.getPattern() << " Quality = " << digi.getQuality() << " Bend = " << digi.getBend()
             << "\n"
             << " KeyHalfStrip = " << digi.getStrip() << " KeyWireGroup = " << digi.getKeyWG()
             << " Type (SIM) = " << digi.getType() << " MPC Link = " << digi.getMPCLink();
}
