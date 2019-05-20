//
// This class keeps the possible non-standard
// status a ROC can have.
//
//
//

#include <cstdint>
#include <set>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"

using namespace std;
using namespace pos;

//======================================================================================
PixelROCStatus::PixelROCStatus() : bits_(0) {}

//======================================================================================
PixelROCStatus::PixelROCStatus(const std::set<ROCstatus>& stat) {
  std::set<ROCstatus>::const_iterator i = stat.begin();

  for (; i != stat.end(); ++i) {
    set(*i);
  }
}

//======================================================================================
PixelROCStatus::~PixelROCStatus() {}

//======================================================================================
void PixelROCStatus::set(ROCstatus stat) {
  reset();
  bits_ = bits_ | (1 << stat);
}

//======================================================================================
void PixelROCStatus::clear(ROCstatus stat) { bits_ = bits_ & (0 << stat); }

//======================================================================================
// Added by Dario (March 4th 2008)
void PixelROCStatus::reset(void) { bits_ = 0; }

//======================================================================================
void PixelROCStatus::set(ROCstatus stat, bool mode) {
  reset();
  if (mode) {
    set(stat);
  } else {
    clear(stat);
  }
}

//======================================================================================
bool PixelROCStatus::get(ROCstatus stat) const { return bits_ & (1 << stat); }

//======================================================================================
string PixelROCStatus::statusName(ROCstatus stat) const {
  if (stat == off)
    return "off";
  if (stat == noHits)
    return "noHits";
  if (stat == noInit)
    return "noInit";
  if (stat == noAnalogSignal)
    return "noAnalogSignal";
  assert(0);
  return "";
}

//======================================================================================
// modified by MR on 11-01-2008 15:06:28
string PixelROCStatus::statusName() const {
  string result = "";
  for (ROCstatus istat = off; istat != nStatus; istat = ROCstatus(istat + 1)) {
    if (get(istat)) {
      result += statusName(istat);
    }
  }
  return result;
}

//======================================================================================
void PixelROCStatus::set(const string& statName) {
  if (!statName.empty()) {
    for (ROCstatus istat = off; istat != nStatus; istat = ROCstatus(istat + 1)) {
      if (statName == statusName(istat)) {
        set(istat);
        return;
      }
    }
    cout << "[PixelROCStatus::set()] statName |" << statName << "| is an invalid keyword" << endl;
    ::abort();
  } else {
    reset();
  }
}
