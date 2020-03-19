/** \file
 * 
 *
 * \author M.Schmitt, Northwestern
 */
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <algorithm>
#include <iterator>

using namespace std;

// Constructors
CSCComparatorDigi::CSCComparatorDigi(int strip, int comparator, int timeBinWord)
    : strip_(strip), comparator_(comparator), timeBinWord_(timeBinWord) {}

CSCComparatorDigi::CSCComparatorDigi() : strip_(0), comparator_(0), timeBinWord_(0) {}

// Comparison

bool CSCComparatorDigi::operator==(const CSCComparatorDigi& digi) const {
  if (getStrip() != digi.getStrip())
    return false;
  if (getComparator() != digi.getComparator())
    return false;
  if (getTimeBinWord() != digi.getTimeBinWord())
    return false;
  return true;
}

//@@ If one wanted to order comparator digis how would one want op< to behave?
// I don't know...
// I think LHS < RHS only makes sense if
// i) time(LHS) .eq. time(RHS)
// AND
// ii) strip(LHS) .lt. strip(RHS)
// But I don't see how this can be useful.

bool CSCComparatorDigi::operator<(const CSCComparatorDigi& digi) const {
  bool result = false;
  if (getTimeBin() == digi.getTimeBin()) {
    result = (getStrip() < digi.getStrip());
  }
  return result;
}

// Getters

int CSCComparatorDigi::getTimeBin() const {
  // Find first bin which fired, counting from 0
  uint16_t tbit = 1;
  int tbin = -1;
  for (int i = 0; i < 16; ++i) {
    if (tbit & timeBinWord_) {
      tbin = i;
      break;
    }
    tbit = tbit << 1;
  }
  return tbin;
}

// This definition is consistent with the one used in
// the function CSCCLCTData::add() in EventFilter/CSCRawToDigi
// The halfstrip counts from 0!
int CSCComparatorDigi::getHalfStrip() const { return (getStrip() - 1) * 2 + getComparator(); }

// Return the fractional half-strip
float CSCComparatorDigi::getFractionalStrip() const { return getStrip() + getComparator() * 0.5f - 0.75f; }

std::vector<int> CSCComparatorDigi::getTimeBinsOn() const {
  std::vector<int> tbins;
  uint16_t tbit = timeBinWord_;
  const uint16_t one = 1;
  for (int i = 0; i < 16; ++i) {
    if (tbit & one)
      tbins.push_back(i);
    tbit = tbit >> 1;
    if (tbit == 0)
      break;  // end already if no more bits set
  }
  return tbins;
}

// Setters
//@@ No way to set time word?

void CSCComparatorDigi::setStrip(int strip) { strip_ = strip; }
void CSCComparatorDigi::setComparator(int comparator) { comparator_ = comparator; }

// Output

void CSCComparatorDigi::print() const {
  std::ostringstream ost;
  ost << "CSCComparatorDigi | strip " << getStrip() << " | comparator " << getComparator() << " | first time bin "
      << getTimeBin() << " | time bins on ";
  std::vector<int> tbins = getTimeBinsOn();
  for (unsigned int i = 0; i < tbins.size(); i++) {
    ost << tbins[i] << " ";
  }
  edm::LogVerbatim("CSCDigi") << ost.str();
}

//@@ Doesn't print all time bins
std::ostream& operator<<(std::ostream& o, const CSCComparatorDigi& digi) {
  return o << " " << digi.getStrip() << " " << digi.getComparator() << " " << digi.getTimeBin();
}
