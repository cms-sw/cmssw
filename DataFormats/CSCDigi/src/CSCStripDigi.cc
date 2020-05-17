/** \file
 *
 *
 * \author M.Schmitt, Northwestern
 */
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cstdint>

// Comparison
bool CSCStripDigi::operator==(const CSCStripDigi& digi) const {
  if (getStrip() != digi.getStrip())
    return false;
  if (getADCCounts().size() != digi.getADCCounts().size())
    return false;
  if (getADCCounts() != digi.getADCCounts())
    return false;
  return true;
}

void CSCStripDigi::setADCCounts(const std::vector<int>& vADCCounts) {
  bool badVal = false;
  for (int vADCCount : vADCCounts) {
    if (vADCCount < 1)
      badVal = true;
  }
  if (!badVal) {
    ADCCounts = vADCCounts;
  } else {
    std::vector<int> ZeroCounts(8, 0);
    ADCCounts = ZeroCounts;
  }
}

// Debug
void CSCStripDigi::print() const {
  std::ostringstream ost;
  ost << "CSCStripDigi | strip " << getStrip() << " | ADCCounts ";
  for (int i : getADCCounts()) {
    ost << i << " ";
  }
  ost << " | Overflow ";
  for (unsigned short i : getADCOverflow()) {
    ost << i << " ";
  }
  ost << " | Overlapped ";
  for (unsigned short i : getOverlappedSample()) {
    ost << i << " ";
  }
  ost << " | L1APhase ";
  for (int i = 0; i < (int)getL1APhase().size(); i++) {
    ost << getL1APhase()[i] << " ";
  }
  edm::LogVerbatim("CSCDigi") << ost.str();
}

std::ostream& operator<<(std::ostream& o, const CSCStripDigi& digi) {
  o << " " << digi.getStrip();
  for (int i : digi.getADCCounts()) {
    o << " " << i;
  }
  return o;
}
