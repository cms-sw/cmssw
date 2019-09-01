/** \file
 * 
 *
 * \author N.Terentiev, CMU
 */
#include "DataFormats/CSCDigi/interface/CSCCFEBStatusDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cstdint>

/// Shift and select
int CSCCFEBStatusDigi::ShiftSel(int nmb, int nshift, int nsel) const {
  int tmp = nmb;
  tmp = tmp >> nshift;
  return tmp = tmp & nsel;
}
/// Get SCA Full Condition
std::vector<uint16_t> CSCCFEBStatusDigi::getSCAFullCond() const {
  /*    std::vector<int> vec(4,0);
    vec[0]=ShiftSel(SCAFullCond_,0,15);  // 4-bit FIFO1 word count
    vec[1]=ShiftSel(SCAFullCond_,4,15);  // 4-bit Block Number if Error Code=1
                                         // (CFEB: SCA Capacitors Full)
                                         // 4-bit FIFO3 word count if Error Code=2
                                         // (CFEB: FPGA FIFO full)
    vec[2]=ShiftSel(SCAFullCond_,9,7);   // Error Code
    vec[3]=ShiftSel(SCAFullCond_,12,15); // DDU Code, should be 0xB
    return vec;*/
  return bWords_;
}
/// Get TS_FLAG bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getTS_FLAG() const {
  std::vector<int> vec(contrWords_.size(), 0);
  int nmb;
  for (unsigned int i = 0; i < vec.size(); i++) {
    nmb = contrWords_[i];
    vec[i] = ShiftSel(nmb, 15, 1);
  }
  return vec;
}

/// Get SCA_FULL bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getSCA_FULL() const {
  std::vector<int> vec(contrWords_.size(), 0);
  int nmb;
  for (unsigned int i = 0; i < vec.size(); i++) {
    nmb = contrWords_[i];
    vec[i] = ShiftSel(nmb, 14, 1);
  }
  return vec;
}

/// Get LCT_PHASE bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getLCT_PHASE() const {
  std::vector<int> vec(contrWords_.size(), 0);
  int nmb;
  for (unsigned int i = 0; i < vec.size(); i++) {
    nmb = contrWords_[i];
    vec[i] = ShiftSel(nmb, 13, 1);
  }
  return vec;
}

/// Get L1A_PHASE bit from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getL1A_PHASE() const {
  std::vector<int> vec(contrWords_.size(), 0);
  int nmb;
  for (unsigned int i = 0; i < vec.size(); i++) {
    nmb = contrWords_[i];
    vec[i] = ShiftSel(nmb, 12, 1);
  }
  return vec;
}

/// Get SCA_BLK 4 bit word from SCA Controller data  per each time slice
std::vector<int> CSCCFEBStatusDigi::getSCA_BLK() const {
  std::vector<int> vec(contrWords_.size(), 0);
  int nmb;
  for (unsigned int i = 0; i < vec.size(); i++) {
    nmb = contrWords_[i];
    vec[i] = ShiftSel(nmb, 8, 15);
  }
  return vec;
}

/// Get TRIG_TIME 8 bit word from SCA Controller data  per each time  slice
std::vector<int> CSCCFEBStatusDigi::getTRIG_TIME() const {
  std::vector<int> vec(contrWords_.size(), 0);
  int nmb;
  for (unsigned int i = 0; i < vec.size(); i++) {
    nmb = contrWords_[i];
    vec[i] = ShiftSel(nmb, 0, 255);
  }
  return vec;
}

/// Debug
void CSCCFEBStatusDigi::print() const {
  edm::LogVerbatim("CSCDigi") << "CSC CFEB # : " << getCFEBNmb();

  std::ostringstream ost;
  ost << " SCAFullCond: ";
  if (!getSCAFullCond().empty()) {
    for (size_t i = 0; i < 4; ++i) {
      ost << " " << (getSCAFullCond())[i];
    }
  } else {
    ost << " "
        << "BWORD is not valid";
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " CRC: ";
  for (size_t i = 0; i < getCRC().size(); ++i) {
    ost << " " << (getCRC())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " TS_FLAG: ";
  for (size_t i = 0; i < getTS_FLAG().size(); ++i) {
    ost << " " << (getTS_FLAG())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " SCA_FULL: ";
  for (size_t i = 0; i < getSCA_FULL().size(); ++i) {
    ost << " " << (getSCA_FULL())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " LCT_PHASE: ";
  for (size_t i = 0; i < getLCT_PHASE().size(); ++i) {
    ost << " " << (getLCT_PHASE())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " L1A_PHASE: ";
  for (size_t i = 0; i < getL1A_PHASE().size(); ++i) {
    ost << " " << (getL1A_PHASE())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " SCA_BLK: ";
  for (size_t i = 0; i < getSCA_BLK().size(); ++i) {
    ost << " " << (getSCA_BLK())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();

  ost.clear();
  ost << " TRIG_TIME: ";
  for (size_t i = 0; i < getTRIG_TIME().size(); ++i) {
    ost << " " << (getTRIG_TIME())[i];
  }
  edm::LogVerbatim("CSCDigi") << ost.str();
}

std::ostream& operator<<(std::ostream& o, const CSCCFEBStatusDigi& digi) {
  o << " " << digi.getCFEBNmb() << "\n";
  for (size_t i = 0; i < 4; ++i) {
    o << " " << (digi.getSCAFullCond())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getCRC().size(); ++i) {
    o << " " << (digi.getCRC())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getTS_FLAG().size(); ++i) {
    o << " " << (digi.getTS_FLAG())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getSCA_FULL().size(); ++i) {
    o << " " << (digi.getSCA_FULL())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getLCT_PHASE().size(); ++i) {
    o << " " << (digi.getLCT_PHASE())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getL1A_PHASE().size(); ++i) {
    o << " " << (digi.getL1A_PHASE())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getSCA_BLK().size(); ++i) {
    o << " " << (digi.getSCA_BLK())[i];
  }
  o << "\n";
  for (size_t i = 0; i < digi.getTRIG_TIME().size(); ++i) {
    o << " " << (digi.getTRIG_TIME())[i];
  }
  o << "\n";

  return o;
}
