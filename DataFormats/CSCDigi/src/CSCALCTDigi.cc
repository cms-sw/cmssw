/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives.
 *
 *
 * \author N. Terentiev, CMU
 */

#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

enum Pattern_Info { NUM_LAYERS = 6, ALCT_PATTERN_WIDTH = 5 };

using namespace std;

/// Constructors
CSCALCTDigi::CSCALCTDigi(const uint16_t valid,
                         const uint16_t quality,
                         const uint16_t accel,
                         const uint16_t patternb,
                         const uint16_t keywire,
                         const uint16_t bx,
                         const uint16_t trknmb,
                         const uint16_t hmt,
                         const Version version)
    : valid_(valid),
      quality_(quality),
      accel_(accel),
      patternb_(patternb),
      keywire_(keywire),
      bx_(bx),
      trknmb_(trknmb),
      hmt_(hmt),
      version_(version) {
  hits_.resize(NUM_LAYERS);
  for (auto& p : hits_) {
    p.resize(ALCT_PATTERN_WIDTH);
  }
}

/// Default
CSCALCTDigi::CSCALCTDigi() {
  clear();  // set contents to zero
}

/// Clears this ALCT.
void CSCALCTDigi::clear() {
  valid_ = 0;
  quality_ = 0;
  accel_ = 0;
  patternb_ = 0;
  keywire_ = 0;
  bx_ = 0;
  trknmb_ = 0;
  fullbx_ = 0;
  hmt_ = 0;
  hits_.resize(NUM_LAYERS);
  version_ = Version::Legacy;
  for (auto& p : hits_) {
    p.resize(ALCT_PATTERN_WIDTH);
  }
}

uint16_t CSCALCTDigi::getHMT() const { return (isRun3() ? hmt_ : std::numeric_limits<uint16_t>::max()); }

void CSCALCTDigi::setHMT(const uint16_t h) { hmt_ = isRun3() ? h : std::numeric_limits<uint16_t>::max(); }

void CSCALCTDigi::setRun3(const bool isRun3) { version_ = isRun3 ? Version::Run3 : Version::Legacy; }

bool CSCALCTDigi::operator>(const CSCALCTDigi& rhs) const {
  bool returnValue = false;

  // Early ALCTs are always preferred to the ones found at later bx's.
  if (getBX() < rhs.getBX()) {
    returnValue = true;
  }
  if (getBX() != rhs.getBX()) {
    return returnValue;
  }

  // The > operator then checks the quality of ALCTs.
  // If two qualities are equal, the ALCT furthest from the beam axis
  // (lowest eta, highest wire group number) is selected.
  uint16_t quality1 = getQuality();
  uint16_t quality2 = rhs.getQuality();
  if (quality1 > quality2) {
    returnValue = true;
  } else if (quality1 == quality2 && getKeyWG() > rhs.getKeyWG()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCALCTDigi::operator==(const CSCALCTDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && getQuality() == rhs.getQuality() && getAccelerator() == rhs.getAccelerator() &&
      getCollisionB() == rhs.getCollisionB() && getKeyWG() == rhs.getKeyWG() && getBX() == rhs.getBX()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCALCTDigi::operator!=(const CSCALCTDigi& rhs) const {
  // True if == is false.
  bool returnValue = true;
  if ((*this) == rhs)
    returnValue = false;
  return returnValue;
}

/// Debug
void CSCALCTDigi::print() const {
  if (isValid()) {
    edm::LogVerbatim("CSCDigi") << "CSC ALCT #" << setw(1) << getTrknmb() << ": Valid = " << setw(1) << isValid()
                                << " Quality = " << setw(2) << getQuality() << " Accel. = " << setw(1)
                                << getAccelerator() << " PatternB = " << setw(1) << getCollisionB()
                                << " Key wire group = " << setw(3) << getKeyWG() << " BX = " << setw(2) << getBX()
                                << " Full BX = " << std::setw(1) << getFullBX();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid Anode LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCALCTDigi& digi) {
  return o << "CSC ALCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " Accel. = " << digi.getAccelerator() << " PatternB = " << digi.getCollisionB()
           << " Key wire group = " << digi.getKeyWG() << " BX = " << digi.getBX();
}
