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
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(const uint16_t iTrackNumber,
                                           const uint16_t ivalid,
                                           const uint16_t iquality,
                                           const uint16_t ikeyWireGroup,
                                           const uint16_t istrip,
                                           const uint16_t ipattern,
                                           const uint16_t ibend,
                                           const uint16_t ibx,
                                           const uint16_t impcLink,
                                           const uint16_t ibx0,
                                           const uint16_t isyncErr,
                                           const uint16_t icscID,
                                           const uint16_t ihmt,
                                           const Version version)
    : trackNumber_(iTrackNumber),
      valid_(ivalid),
      quality_(iquality),
      keyWireGroup_(ikeyWireGroup),
      strip_(istrip),
      pattern_(ipattern),
      bend_(ibend),
      bx_(ibx),
      mpcLink_(impcLink),
      bx0_(ibx0),
      syncErr_(isyncErr),
      cscID_(icscID),
      hmt_(ihmt),
      version_(version) {}

/// Default
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi() {
  clear();  // set contents to zero
  version_ = Version::Legacy;
}

/// Clears this LCT.
void CSCCorrelatedLCTDigi::clear() {
  trackNumber_ = 0;
  valid_ = 0;
  quality_ = 0;
  keyWireGroup_ = 0;
  strip_ = 0;
  pattern_ = 0;
  bend_ = 0;
  bx_ = 0;
  mpcLink_ = 0;
  bx0_ = 0;
  syncErr_ = 0;
  cscID_ = 0;
  hmt_ = 0;
}

uint16_t CSCCorrelatedLCTDigi::strip(const uint16_t n) const {
  // all 10 bits
  if (n == 8) {
    return 2 * strip(4) + eightStrip();
  }
  // lowest 9 bits
  else if (n == 4) {
    return 2 * strip(2) + quartStrip();
  }
  // lowest 8 bits
  else {
    return strip_ & kHalfStripMask;
  }
}

void CSCCorrelatedLCTDigi::setQuartStrip(const bool quartStrip) {
  // clear the old value
  strip_ &= ~(kQuartStripMask << kQuartStripShift);

  // set the new value
  strip_ |= quartStrip << kQuartStripShift;
}

void CSCCorrelatedLCTDigi::setEightStrip(const bool eightStrip) {
  // clear the old value
  strip_ &= ~(kEightStripMask << kEightStripShift);

  // set the new value
  strip_ |= eightStrip << kEightStripShift;
}

bool CSCCorrelatedLCTDigi::quartStrip() const { return (strip_ >> kQuartStripShift) & kQuartStripMask; }

bool CSCCorrelatedLCTDigi::eightStrip() const { return (strip_ >> kEightStripShift) & kEightStripMask; }

/// return the fractional strip
float CSCCorrelatedLCTDigi::fractionalStrip(const uint16_t n) const {
  if (n == 8) {
    return 0.125f * (strip() + 1) - 0.0625f;
  } else if (n == 4) {
    return 0.25f * (strip() + 1) - 0.125f;
  } else {
    return 0.5f * (strip() + 1) - 0.25f;
  }
}

uint16_t CSCCorrelatedLCTDigi::clctPattern() const {
  return (isRun3() ? std::numeric_limits<uint16_t>::max() : (pattern_ & 0xF));
}

uint16_t CSCCorrelatedLCTDigi::hmt() const { return (isRun3() ? hmt_ : std::numeric_limits<uint16_t>::max()); }

void CSCCorrelatedLCTDigi::setHMT(const uint16_t h) { hmt_ = isRun3() ? h : std::numeric_limits<uint16_t>::max(); }

void CSCCorrelatedLCTDigi::setRun3(const bool isRun3) { version_ = isRun3 ? Version::Run3 : Version::Legacy; }

/// Comparison
bool CSCCorrelatedLCTDigi::operator==(const CSCCorrelatedLCTDigi& rhs) const {
  return ((trackNumber_ == rhs.trackNumber_) && (quality_ == rhs.quality_) && (keyWireGroup_ == rhs.keyWireGroup_) &&
          (strip_ == rhs.strip_) && (pattern_ == rhs.pattern_) && (bend_ == rhs.bend_) && (bx_ == rhs.bx_) &&
          (valid_ == rhs.valid_) && (mpcLink_ == rhs.mpcLink_) && (hmt_ == rhs.hmt_));
}

/// Debug
void CSCCorrelatedLCTDigi::print() const {
  if (isValid()) {
    edm::LogVerbatim("CSCDigi") << "CSC LCT #" << trackNumber() << ": Valid = " << isValid()
                                << " Quality = " << quality() << " Key Wire = " << keyWireGroup()
                                << " Strip = " << strip() << " Pattern = " << pattern()
                                << " Bend = " << ((bend() == 0) ? 'L' : 'R') << " BX = " << bx()
                                << " MPC Link = " << mpcLink() << " HMT Bit = " << hmt();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid correlated LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCCorrelatedLCTDigi& digi) {
  return o << "CSC LCT #" << digi.trackNumber() << ": Valid = " << digi.isValid() << " Quality = " << digi.quality()
           << " MPC Link = " << digi.mpcLink() << " cscID = " << digi.cscID() << "\n"
           << "  cathode info: Strip = " << digi.strip() << " Pattern = " << digi.pattern()
           << " Bend = " << ((digi.bend() == 0) ? 'L' : 'R') << "\n"
           << "    anode info: Key wire = " << digi.keyWireGroup() << " BX = " << digi.bx() << " bx0 = " << digi.bx0()
           << " syncErr = " << digi.syncErr() << " HMT Bit = " << digi.hmt() << "\n";
}
