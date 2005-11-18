/** \file
 * 
 *  $Date: 2005/11/17 13:03:05 $
 *  $Revision: 1.1 $
 *
 * \author M.Schmitt, Northwestern
 */
#include <DataFormats/CSCDigi/interface/CSCStripDigi.h>
#include <iostream>
#include <bitset>
#include <vector>

using namespace std;

// Constructors
CSCStripDigi::CSCStripDigi (int strip, vector<int> ADCCounts){
  set(strip, ADCCounts);
}

CSCStripDigi::CSCStripDigi (){
  vector<int> ZeroCounts(8,0);
  set(0,ZeroCounts);
}

// Copy constructor
CSCStripDigi::CSCStripDigi(const CSCStripDigi& digi) {
  aStripDigi = digi.aStripDigi;
}

// Assignment
CSCStripDigi& 
CSCStripDigi::operator=(const CSCStripDigi& digi){
  aStripDigi = digi.aStripDigi;
  return *this;
}

// Comparison
bool
CSCStripDigi::operator == (const CSCStripDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getADCCounts().size() != digi.getADCCounts().size() ) return false;
  if ( getADCCounts() != digi.getADCCounts() ) return false;
  return true;
}

// Getters
int CSCStripDigi::getStrip() const { return data()->strip; }
std::vector<int> CSCStripDigi::getADCCounts() const { return data()->ADCCounts; }

// Setters
void CSCStripDigi::setStrip(int strip) {
  data()->strip = strip;
}
void CSCStripDigi::setADCCounts(vector<int>ADCCounts) {
  bool badVal = false;
  for (int i=0; i<(int)ADCCounts.size(); i++) {
    if (ADCCounts[i] < 1) badVal = true;
  }
  if ( !badVal ) {
    data()->ADCCounts = ADCCounts;
  } else {
    vector<int> ZeroCounts(8,0);
    data()->ADCCounts = ZeroCounts;
  }
}

// Debug
void
CSCStripDigi::print() const {
  //  cout << "CSC Strip: " << strip() << " ADC Counts: ";
  //  for (int i=0; i<ADCCounts().size(); i++) {cout << ADCCounts[i] << " ";}
  cout << "\n";
}

void
CSCStripDigi::dump() const {
  // Do we need this?
}

// ----- Private members
void
CSCStripDigi::set(int strip, vector<int> ADCCounts) {
  data()->strip = strip;
  data()->ADCCounts = ADCCounts;
}

CSCStripDigi::theStripDigi*
CSCStripDigi::data() {
  return reinterpret_cast<theStripDigi*>(&aStripDigi);
}

const CSCStripDigi::theStripDigi*
CSCStripDigi::data() const {
  return reinterpret_cast<theStripDigi*>(&aStripDigi);
}

void CSCStripDigi::setData(theStripDigi p){
  *(data()) = p;
}
