/** \file
 *
 *  $Date: 2008/10/29 18:34:41 $
 *  $Revision: 1.16 $
 *
 * \author M.Schmitt, Northwestern
 */
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include <iostream>
#include <stdint.h>

// Constructors
CSCStripDigi::CSCStripDigi (const int & istrip, const std::vector<int> & vADCCounts, const std::vector<uint16_t> & vADCOverflow,
			    const std::vector<uint16_t> & vOverlap, const std::vector<uint16_t> & vErrorstat ):
  strip(istrip),
  ADCCounts(vADCCounts),
  ADCOverflow(vADCOverflow),
  OverlappedSample(vOverlap),
  Errorstat(vErrorstat)
{
}

CSCStripDigi::CSCStripDigi (const int & istrip, const std::vector<int> & vADCCounts):
  strip(istrip),
  ADCCounts(vADCCounts),
  ADCOverflow(8,0),
  OverlappedSample(8,0),
  Errorstat(8,0)
{
}


CSCStripDigi::CSCStripDigi ():
  strip(0),
  ADCCounts(8,0),
  ADCOverflow(8,0),
  OverlappedSample(8,0),
  Errorstat(8,0)
{
}

std::vector<int> CSCStripDigi::getADCCounts() const { return ADCCounts; }

// Comparison
bool
CSCStripDigi::operator == (const CSCStripDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getADCCounts().size() != digi.getADCCounts().size() ) return false;
  if ( getADCCounts() != digi.getADCCounts() ) return false;
  return true;
}

// Getters
//int CSCStripDigi::getStrip() const { return strip; }
//std::vector<int> CSCStripDigi::getADCCounts() const { return ADCCounts; }


// Setters
//void CSCStripDigi::setStrip(int istrip) {
//  strip = istrip;
//}


void CSCStripDigi::setADCCounts(std::vector<int>vADCCounts) {
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
}

std::ostream & operator<<(std::ostream & o, const CSCStripDigi& digi) {
  o << " " << digi.getStrip();
  for (size_t i = 0; i<digi.getADCCounts().size(); ++i ){
    o <<" " <<(digi.getADCCounts())[i]; }
  return o;

}



