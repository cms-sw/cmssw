/** \file
 * 
 *  $Date: 2006/04/06 11:18:37 $
 *  $Revision: 1.5 $
 *
 * \author M.Schmitt, Northwestern
 */
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <iostream>
#include <bitset>

using namespace std;

// Constructors
CSCComparatorDigi::CSCComparatorDigi (int istrip, int icomparator, int itimeBin){
  strip = istrip;
  comparator = icomparator;
  timeBin = itimeBin;
}


CSCComparatorDigi::CSCComparatorDigi (){
  strip = comparator = timeBin =0;
}


// Comparison
bool
CSCComparatorDigi::operator == (const CSCComparatorDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getComparator() != digi.getComparator() ) return false;
  return true;
}


bool 
CSCComparatorDigi::operator<(const CSCComparatorDigi& digi) const {
  bool result = true;
  // sort on time first, then strip
  if(getTimeBin() == digi.getTimeBin()) {
    result = (getStrip() < digi.getStrip());
  }
  else {
    result = (getTimeBin() == digi.getTimeBin());
  }
  return result;
}


// Getters
int CSCComparatorDigi::getStrip() const { return strip; }
int CSCComparatorDigi::getComparator() const { return comparator; }
int CSCComparatorDigi::getTimeBin() const {return timeBin; }

// Setters
void CSCComparatorDigi::setStrip(int istrip) {
  strip = istrip;
}
void CSCComparatorDigi::setComparator(int icomparator) {
  comparator = icomparator;
}

// Debug
void
CSCComparatorDigi::print() const {
  cout << "CSC Comparator strip: " << getStrip() 
       << " Comparator: " << getComparator() 
       << " Time Bin: "<< getTimeBin() << endl;
}


