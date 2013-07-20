/** \file
 * 
 *  $Date: 2009/04/24 18:01:44 $
 *  $Revision: 1.9 $
 *
 * \author M.Schmitt, Northwestern
 */
#include <DataFormats/CSCDigi/interface/CSCComparatorDigi.h>
#include <iostream>
#include <algorithm>
#include <iterator>

using namespace std;

// Constructors
CSCComparatorDigi::CSCComparatorDigi( int strip, int comparator, int timeBinWord )
  : strip_( strip ), comparator_( comparator ), timeBinWord_( timeBinWord ) {
}


CSCComparatorDigi::CSCComparatorDigi() 
  : strip_( 0 ), comparator_( 0 ), timeBinWord_( 0 ) {
}


// Comparison

bool
CSCComparatorDigi::operator == (const CSCComparatorDigi& digi) const {
  if ( getStrip() != digi.getStrip() ) return false;
  if ( getComparator() != digi.getComparator() ) return false;
  if ( getTimeBinWord() != digi.getTimeBinWord() ) return false;
  return true;
}


//@@ If one wanted to order comparator digis how would one want op< to behave?
// I don't know...
// I think LHS < RHS only makes sense if
// i) time(LHS) .eq. time(RHS)
// AND
// ii) strip(LHS) .lt. strip(RHS)
// But I don't see how this can be useful.

bool 
CSCComparatorDigi::operator<(const CSCComparatorDigi& digi) const {
  bool result = false;
  if(getTimeBin() == digi.getTimeBin()) {
    result = (getStrip() < digi.getStrip());
  }
  return result;
}


// Getters

int CSCComparatorDigi::getTimeBin() const {
  // Find first bin which fired, counting from 0
  uint16_t tbit=1;
  int tbin=-1;
  for(int i=0;i<16;++i) {
    if(tbit & timeBinWord_) {
      tbin=i;
      break;
    }
    tbit=tbit<<1;
  }
  return tbin;
}

std::vector<int> CSCComparatorDigi::getTimeBinsOn() const {
  std::vector<int> tbins;
  uint16_t tbit = timeBinWord_;
  const uint16_t one=1;
  for(int i=0;i<16;++i) {
    if(tbit & one) tbins.push_back(i);
    tbit=tbit>>1;
    if(tbit==0) break; // end already if no more bits set
  }
  return tbins;                                  
}

// Setters
//@@ No way to set time word?

void CSCComparatorDigi::setStrip(int strip) {
  strip_ = strip;
}
void CSCComparatorDigi::setComparator(int comparator) {
  comparator_ = comparator;
}

// Output

void
CSCComparatorDigi::print() const {
  std::cout << "CSCComparatorDigi strip: " << getStrip() 
       << " comparator: " << getComparator() 
	    << " first time bin: "<< getTimeBin()
       << " time bins on: ";
  std::vector<int> tbins=getTimeBinsOn();
  std::copy( tbins.begin(), tbins.end(), 
     std::ostream_iterator<int>( std::cout, " "));
  std::cout << std::endl; 
}

//@@ Doesn't print all time bins
std::ostream & operator<<(std::ostream & o, const CSCComparatorDigi& digi) {
  return o << " " << digi.getStrip()
	   << " " << digi.getComparator()
	   << " " << digi.getTimeBin();
}  


