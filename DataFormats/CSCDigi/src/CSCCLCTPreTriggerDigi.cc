#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>
#include <iostream>

/// Constructors
CSCCLCTPreTriggerDigi::CSCCLCTPreTriggerDigi(const int valid,
                                             const int quality,
                                             const int pattern,
                                             const int striptype,
                                             const int bend,
                                             const int strip,
                                             const int cfeb,
                                             const int bx,
                                             const int trknmb,
                                             const int fullbx)
    : valid_(valid),
      quality_(quality),
      pattern_(pattern),
      striptype_(striptype),
      bend_(bend),
      strip_(strip),
      cfeb_(cfeb),
      bx_(bx),
      trknmb_(trknmb),
      fullbx_(fullbx) {}

/// Default
CSCCLCTPreTriggerDigi::CSCCLCTPreTriggerDigi()
    : valid_(0),
      quality_(0),
      pattern_(0),
      striptype_(0),
      bend_(0),
      strip_(0),
      cfeb_(0),
      bx_(0),
      trknmb_(0),
      fullbx_(0) {}

/// Clears this CLCT.
void CSCCLCTPreTriggerDigi::clear() {
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
}

bool CSCCLCTPreTriggerDigi::operator>(const CSCCLCTPreTriggerDigi& rhs) const {
  // Several versions of CLCT sorting criteria were used before 2008.
  // They are available in CMSSW versions prior to 3_1_0; here we only keep
  // the latest one, used in TMB-07 firmware (w/o distrips).
  bool returnValue = false;

  int quality1 = quality();
  int quality2 = rhs.quality();
  // The bend-direction bit pid[0] is ignored (left and right bends have
  // equal quality).
  int pattern1 = pattern() & 14;
  int pattern2 = rhs.pattern() & 14;

  // Better-quality CLCTs are preferred.
  // If two qualities are equal, larger pattern id (i.e., straighter pattern)
  // is preferred; left- and right-bend patterns are considered to be of
  // the same quality.
  // If both qualities and pattern id's are the same, lower keystrip
  // is preferred.
  if ((quality1 > quality2) || (quality1 == quality2 && pattern1 > pattern2) ||
      (quality1 == quality2 && pattern1 == pattern2 && keyStrip() < rhs.keyStrip())) {
    returnValue = true;
  }

  return returnValue;
}

bool CSCCLCTPreTriggerDigi::operator==(const CSCCLCTPreTriggerDigi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && quality() == rhs.quality() && pattern() == rhs.pattern() &&
      keyStrip() == rhs.keyStrip() && stripType() == rhs.stripType() && bend() == bend() && bx() == rhs.bx()) {
    returnValue = true;
  }
  return returnValue;
}

bool CSCCLCTPreTriggerDigi::operator!=(const CSCCLCTPreTriggerDigi& rhs) const {
  // True if == is false.
  bool returnValue = true;
  if ((*this) == rhs)
    returnValue = false;
  return returnValue;
}

/// Debug
void CSCCLCTPreTriggerDigi::print() const {
  if (isValid()) {
    char stripTypeChar = (stripType() == 0) ? 'D' : 'H';
    char bendChar = (bend() == 0) ? 'L' : 'R';

    edm::LogVerbatim("CSCDigi") << " CSC CLCT #" << std::setw(1) << trackNumber() << ": Valid = " << std::setw(1)
                                << isValid() << " Key Strip = " << std::setw(3) << keyStrip()
                                << " Strip = " << std::setw(2) << strip() << " Quality = " << std::setw(1) << quality()
                                << " Pattern = " << std::setw(1) << pattern() << " Bend = " << std::setw(1) << bendChar
                                << " Strip type = " << std::setw(1) << stripTypeChar << " CFEB ID = " << std::setw(1)
                                << cfeb() << " BX = " << std::setw(1) << bx() << " Full BX= " << std::setw(1)
                                << fullBX();
  } else {
    edm::LogVerbatim("CSCDigi") << "Not a valid Cathode LCT.";
  }
}

std::ostream& operator<<(std::ostream& o, const CSCCLCTPreTriggerDigi& digi) {
  return o << "CSC CLCT #" << digi.trackNumber() << ": Valid = " << digi.isValid() << " Quality = " << digi.quality()
           << " Pattern = " << digi.pattern() << " StripType = " << digi.stripType() << " Bend = " << digi.bend()
           << " Strip = " << digi.strip() << " KeyStrip = " << digi.keyStrip() << " CFEB = " << digi.cfeb()
           << " BX = " << digi.bx();
}
