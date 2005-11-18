/** \file
 * 
 *  $Date: 2005/11/17 13:03:05 $
 *  $Revision: 1.1 $
 *
 * \author M.Schmitt, Northwestern
 */
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <iostream>
#include <bitset>

using namespace std;

// Constructors
CSCComparatorDigi::CSCComparatorDigi (int strip, int comparator){
  set(strip, comparator);
}

CSCComparatorDigi::CSCComparatorDigi (){
  set(0,0);
}

// Copy constructor
CSCComparatorDigi::CSCComparatorDigi(const CSCComparatorDigi& digi) {
  aComparatorDigi = digi.aComparatorDigi;
}

// Assignment
CSCComparatorDigi& 
CSCComparatorDigi::operator=(const CSCComparatorDigi& digi){
  aComparatorDigi = digi.aComparatorDigi;
  return *this;
}

// Comparison
bool
CSCComparatorDigi::operator == (const CSCComparatorDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getComparator() != digi.getComparator() ) return false;
  return true;
}

// Getters
int CSCComparatorDigi::getStrip() const { return data()->strip; }
int CSCComparatorDigi::getComparator() const { return data()->comparator; }

// Setters
void CSCComparatorDigi::setStrip(int strip) {
  data()->strip = strip;
}
void CSCComparatorDigi::setComparator(int comparator) {
    data()->comparator = comparator;
}

// Debug
void
CSCComparatorDigi::print() const {
  cout << "CSC Comparator strip: " << getStrip() 
       << " Comparator: " << getComparator() << endl;
}

void
CSCComparatorDigi::dump() const {
  // Do we need this?
}

// ----- Private members
void
CSCComparatorDigi::set(int strip, int comparator) {
  data()->strip = strip;
  data()->comparator = comparator;
}

