#include "DataFormats/CSCDigi/interface/CSCALCTPreTriggerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

using namespace std;

/// Constructors
CSCALCTPreTriggerDigi::CSCALCTPreTriggerDigi(const int valid,
                                             const int quality,
                                             const int accel,
                                             const int patternb,
                                             const int keywire,
                                             const int bx,
                                             const int trknmb) {
  valid_ = valid;
  quality_ = quality;
  accel_ = accel;
  patternb_ = patternb;
  keywire_ = keywire;
  bx_ = bx;
  trknmb_ = trknmb;
}

/// Default
CSCALCTPreTriggerDigi::CSCALCTPreTriggerDigi() {
  clear();  // set contents to zero
}

/// Clears this ALCT.
void CSCALCTPreTriggerDigi::clear() {
  valid_ = 0;
  quality_ = 0;
  accel_ = 0;
  patternb_ = 0;
  keywire_ = 0;
  bx_ = 0;
  trknmb_ = 0;
  fullbx_ = 0;
}

bool CSCALCTPreTriggerDigi::operator>(const CSCALCTPreTriggerDigi& rhs) const {
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
  int quality1 = getQuality();
  int quality2 = rhs.getQuality();
  if (quality1 > quality2) {
    returnValue = true;
  } else if (quality1 == quality2 && getKeyWG() > rhs.getKeyWG()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCALCTPreTriggerDigi::operator==(const CSCALCTPreTriggerDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && getQuality() == rhs.getQuality() && getAccelerator() == rhs.getAccelerator() &&
      getCollisionB() == rhs.getCollisionB() && getKeyWG() == rhs.getKeyWG() && getBX() == rhs.getBX()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCALCTPreTriggerDigi::operator!=(const CSCALCTPreTriggerDigi& rhs) const {
  // True if == is false.
  bool returnValue = true;
  if ((*this) == rhs)
    returnValue = false;
  return returnValue;
}

/// Debug
void CSCALCTPreTriggerDigi::print() const {
  if (isValid()) {
    edm::LogVerbatim("CSCDigi") << "CSC ALCT #" << setw(1) << getTrknmb() << ": Valid = " << setw(1) << isValid()
                                << " Quality = " << setw(2) << getQuality() << " Accel. = " << setw(1)
                                << getAccelerator() << " PatternB = " << setw(1) << getCollisionB()
                                << " Key wire group = " << setw(3) << getKeyWG() << " BX = " << setw(2) << getBX()
                                << " Full BX= " << std::setw(1) << getFullBX();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid Anode LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCALCTPreTriggerDigi& digi) {
  return o << "CSC ALCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " Accel. = " << digi.getAccelerator() << " PatternB = " << digi.getCollisionB()
           << " Key wire group = " << digi.getKeyWG() << " BX = " << digi.getBX();
}
