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
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(const int itrknmb,
                                           const int ivalid,
                                           const int iquality,
                                           const int ikeywire,
                                           const int istrip,
                                           const int ipattern,
                                           const int ibend,
                                           const int ibx,
                                           const int impclink,
                                           const uint16_t ibx0,
                                           const uint16_t isyncErr,
                                           const uint16_t icscID)
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
      cscID(icscID) {}

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
}

/// return the fractional strip
float CSCCorrelatedLCTDigi::getFractionalStrip() const { return 0.5f * (getStrip() + 1) - 0.25f; }

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
                                << " MPC Link = " << getMPCLink();
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
           << " syncErr = " << digi.getSyncErr() << "\n";
}
