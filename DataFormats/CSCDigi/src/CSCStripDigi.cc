/** \file
 *
 *
 * \author M.Schmitt, Northwestern
 */
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include <iostream>
#include <cstdint>


// Comparison
bool
CSCStripDigi::operator == (const CSCStripDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getADCCounts().size() != digi.getADCCounts().size() ) return false;
  if ( getADCCounts() != digi.getADCCounts() ) return false;
  return true;
}



void CSCStripDigi::setADCCounts(const std::vector<int>&vADCCounts) {
  bool badVal = false;
  for (int i=0; i<(int)vADCCounts.size(); i++) {
    if (vADCCounts[i] < 1) badVal = true;
  }
  if ( !badVal ) {
    ADCCounts = vADCCounts;
  } else {
    std::vector<int> ZeroCounts(8,0);
    ADCCounts = ZeroCounts;
  }
}

// Debug
void
CSCStripDigi::print() const {
  std::cout << "CSC Strip: " << getStrip() << "  ADC Counts: ";
  for (int i=0; i<(int)getADCCounts().size(); i++) {std::cout << getADCCounts()[i] << " ";}
  std::cout << "\n";
  std::cout << "            " << "  ADCOverflow: ";
  for (int i=0; i<(int)getADCOverflow().size(); i++) {std::cout << getADCOverflow()[i] << " ";}
  std::cout << "\n";
  std::cout << "            " << "  OverflappedSample: ";
  for (int i=0; i<(int)getOverlappedSample().size(); i++) {
  //if(getOverlappedSample()[i]!=1)
  std::cout << getOverlappedSample()[i] << " ";}
  std::cout << "\n";
  std::cout << "            " << "  L1APhases: ";
  for(int i=0; i<(int)getL1APhase().size(); i++){
     std::cout << getL1APhase()[i] << " ";
  }
  std::cout << "\n";
}

std::ostream & operator<<(std::ostream & o, const CSCStripDigi& digi) {
  o << " " << digi.getStrip();
  for (size_t i = 0; i<digi.getADCCounts().size(); ++i ){
    o <<" " <<(digi.getADCCounts())[i]; }
  return o;

}



