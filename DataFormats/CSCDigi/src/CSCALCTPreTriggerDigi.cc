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
  if (bx() < rhs.bx()) {
    returnValue = true;
  }
  if (bx() != rhs.bx()) {
    return returnValue;
  }

  // The > operator then checks the quality of ALCTs.
  // If two qualities are equal, the ALCT furthest from the beam axis
  // (lowest eta, highest wire group number) is selected.
  int quality1 = quality();
  int quality2 = rhs.quality();
  if (quality1 > quality2) {
    returnValue = true;
  } else if (quality1 == quality2 && keyWireGroup() > rhs.keyWireGroup()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCALCTPreTriggerDigi::operator==(const CSCALCTPreTriggerDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && quality() == rhs.quality() && accelerator() == rhs.accelerator() &&
      collisionB() == rhs.collisionB() && keyWireGroup() == rhs.keyWireGroup() && bx() == rhs.bx()) {
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
    edm::LogVerbatim("CSCDigi") << "CSC ALCT #" << setw(1) << trackNumber() << ": Valid = " << setw(1) << isValid()
                                << " Quality = " << setw(2) << quality() << " Accel. = " << setw(1) << accelerator()
                                << " PatternB = " << setw(1) << collisionB() << " Key wire group = " << setw(3)
                                << keyWireGroup() << " BX = " << setw(2) << bx() << " Full BX= " << std::setw(1)
                                << fullBX();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid Anode LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCALCTPreTriggerDigi& digi) {
  return o << "CSC ALCT #" << digi.trackNumber() << ": Valid = " << digi.isValid() << " Quality = " << digi.quality()
           << " Accel. = " << digi.accelerator() << " PatternB = " << digi.collisionB()
           << " Key wire group = " << digi.keyWireGroup() << " BX = " << digi.bx();
}
