#include "DataFormats/CSCDigi/interface/CSCCLCTRun3Digi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

/// Constructors
CSCCLCTRun3Digi::CSCCLCTRun3Digi(const int valid,
                                 const int quality,
                                 const int pattern,
                                 const int striptype,
                                 const int bend,
                                 const int strip,
                                 const int cfeb,
                                 const int bx,
                                 const int compCode,
                                 const int trknmb,
                                 const int fullbx)
    : CSCCLCTDigi(valid, quality, pattern, striptype, bend, strip, cfeb, bx, trknmb, fullbx), compCode_(compCode) {}

/// Default
CSCCLCTRun3Digi::CSCCLCTRun3Digi() : CSCCLCTDigi(), compCode_(0) {
  hits_.resize(6);
  for (auto& p : hits_) {
    p.resize(11);
  }
}

/// Clears this CLCT.
void CSCCLCTRun3Digi::clear() {
  CSCCLCTDigi::clear();
  compCode_ = 0;
  hits_.clear();
}

int CSCCLCTRun3Digi::getKeyStrip(bool doFit) const {
  if (!doFit)
    return CSCCLCTDigi::getKeyStrip();
  int keyStrip = cfeb_ * 64 + strip_;
  return keyStrip;
}

bool CSCCLCTRun3Digi::operator==(const CSCCLCTRun3Digi& rhs) const {
  // Exact equality.
  bool returnValue = false;
  if (isValid() == rhs.isValid() && getQuality() == rhs.getQuality() && getPattern() == rhs.getPattern() &&
      getKeyStrip() == rhs.getKeyStrip() && getStripType() == rhs.getStripType() && getBend() == getBend() &&
      getBX() == rhs.getBX() && getCompCode() == rhs.getCompCode()) {
    returnValue = true;
  }
  return returnValue;
}

/// Debug
void CSCCLCTRun3Digi::print() const {
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

std::ostream& operator<<(std::ostream& o, const CSCCLCTRun3Digi& digi) {
  return o << "CSC CLCT #" << digi.getTrknmb() << ": Valid = " << digi.isValid() << " Quality = " << digi.getQuality()
           << " Pattern = " << digi.getPattern() << " StripType = " << digi.getStripType()
           << " Bend = " << digi.getBend() << " Strip = " << digi.getStrip() << " KeyStrip = " << digi.getKeyStrip()
           << " CFEB = " << digi.getCFEB() << " BX = " << digi.getBX() << " Comp Code " << digi.getCompCode();
}
