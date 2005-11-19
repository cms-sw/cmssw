/** \file
 * 
 *  $Date: 2005/11/18 19:22:37 $
 *  $Revision: 1.2 $
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

CSCComparatorDigi::CSCComparatorDigi (theComparatorDigi aComparatorDigi){
  setData(aComparatorDigi);
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
  typedef bitset<8*sizeof(theComparatorDigi)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

// ----- Private members
void
CSCComparatorDigi::set(int strip, int comparator) {
  data()->strip = strip;
  data()->comparator = comparator;
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
