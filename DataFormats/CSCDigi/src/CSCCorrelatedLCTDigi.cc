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
  version_ = Version::Legacy;
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
}

uint16_t CSCCorrelatedLCTDigi::getStrip(const uint16_t n) const {
  // all 10 bits
  if (n == 8) {
    return 2 * getStrip(4) + getEightStrip();
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
  // clear the old value
  strip &= ~(kQuartStripMask << kQuartStripShift);

  // set the new value
  strip |= quartStrip << kQuartStripShift;
}

void CSCCorrelatedLCTDigi::setEightStrip(const bool eightStrip) {
  // clear the old value
  strip &= ~(kEightStripMask << kEightStripShift);

  // set the new value
  strip |= eightStrip << kEightStripShift;
}

bool CSCCorrelatedLCTDigi::getQuartStrip() const { return (strip >> kQuartStripShift) & kQuartStripMask; }

bool CSCCorrelatedLCTDigi::getEightStrip() const { return (strip >> kEightStripShift) & kEightStripMask; }

/// return the fractional strip
float CSCCorrelatedLCTDigi::getFractionalStrip(const uint16_t n) const {
  if (n == 8) {
    return 0.125f * (getStrip() + 1) - 0.0625f;
  } else if (n == 4) {
    return 0.25f * (getStrip() + 1) - 0.125f;
  } else {
    return 0.5f * (getStrip() + 1) - 0.25f;
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
          (mpclink == rhs.mpclink) && (hmt == rhs.hmt));
}

/// Debug
void CSCCorrelatedLCTDigi::print() const {
  if (isValid()) {
    edm::LogVerbatim("CSCDigi") << "CSC LCT #" << getTrknmb() << ": Valid = " << isValid()
                                << " Quality = " << getQuality() << " Key Wire = " << getKeyWG()
                                << " Strip = " << getStrip() << " Pattern = " << getPattern()
                                << " Bend = " << ((getBend() == 0) ? 'L' : 'R') << " BX = " << getBX()
                                << " MPC Link = " << getMPCLink() << " HMT Bit = " << getHMT();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid correlated LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCCorrelatedLCTDigi& digi) {
  return o << "CSC LCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " MPC Link = " << digi.getMPCLink() << " cscID = " << digi.getCSCID() << "\n"
           << "  cathode info: Strip = " << digi.getStrip() << " Pattern = " << digi.getPattern()
           << " Bend = " << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           << "    anode info: Key wire = " << digi.getKeyWG() << " BX = " << digi.getBX() << " bx0 = " << digi.getBX0()
           << " syncErr = " << digi.getSyncErr() << " HMT Bit = " << digi.getHMT() << "\n";
}
