/** \file
 * 
 *  $Date: 2005/11/19 13:58:18 $
 *  $Revision: 1.3 $
 *
 * \author M.Schmitt, Northwestern
 */
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <iostream>
#include <bitset>

using namespace std;

// Constructors
CSCComparatorDigi::CSCComparatorDigi (int strip, int comparator, int timeBin){
  set(strip, comparator, timeBin);
}

CSCComparatorDigi::CSCComparatorDigi (theComparatorDigi aComparatorDigi){
  setData(aComparatorDigi);
}

CSCComparatorDigi::CSCComparatorDigi (){
  set(0,0, 0);
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
int CSCComparatorDigi::getStrip() const { return data()->strip; }
int CSCComparatorDigi::getComparator() const { return data()->comparator; }
int CSCComparatorDigi::getTimeBin() const {return data()->timeBin; }

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
  typedef bitset<8*sizeof(theComparatorDigi)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

// ----- Private members
void
CSCComparatorDigi::set(int strip, int comparator, int timeBin) {
  data()->strip = strip;
  data()->comparator = comparator;
  data()->timeBin = timeBin;
}

CSCComparatorDigi::theComparatorDigi*
CSCComparatorDigi::data() {
  return reinterpret_cast<theComparatorDigi*>(&aComparatorDigi);
}

const CSCComparatorDigi::theComparatorDigi*
CSCComparatorDigi::data() const {
  return reinterpret_cast<const theComparatorDigi*>(&aComparatorDigi);
}

void CSCComparatorDigi::setData(theComparatorDigi p){
  *(data()) = p;
}
