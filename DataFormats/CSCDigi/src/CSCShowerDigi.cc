#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

using namespace std;

/// Constructors
CSCShowerDigi::CSCShowerDigi(const uint16_t bitsInTime, const uint16_t bitsOutTime, const uint16_t cscID)
    : cscID_(cscID) {
  setDataWord(bitsInTime, bits_, kInTimeShift, kInTimeMask);
  setDataWord(bitsOutTime, bits_, kOutTimeShift, kOutTimeMask);
}

/// Default
CSCShowerDigi::CSCShowerDigi() : bits_(0), cscID_(0) {}

bool CSCShowerDigi::isValid() const {
  // any loose shower is valid
  return isLooseInTime() or isLooseOutTime();
}

bool CSCShowerDigi::isLooseInTime() const { return bitsInTime() >= kLoose; }

bool CSCShowerDigi::isNominalInTime() const { return bitsInTime() >= kNominal; }

bool CSCShowerDigi::isTightInTime() const { return bitsInTime() >= kTight; }

bool CSCShowerDigi::isLooseOutTime() const { return bitsOutTime() >= kLoose; }

bool CSCShowerDigi::isNominalOutTime() const { return bitsOutTime() >= kNominal; }

bool CSCShowerDigi::isTightOutTime() const { return bitsOutTime() >= kTight; }

uint16_t CSCShowerDigi::bitsInTime() const { return getDataWord(bits_, kInTimeShift, kInTimeMask); }

uint16_t CSCShowerDigi::bitsOutTime() const { return getDataWord(bits_, kOutTimeShift, kOutTimeMask); }

void CSCShowerDigi::setDataWord(const uint16_t newWord, uint16_t& word, const unsigned shift, const unsigned mask) {
  // clear the old value
  word &= ~(mask << shift);

  // set the new value
  word |= newWord << shift;
}

uint16_t CSCShowerDigi::getDataWord(const uint16_t word, const unsigned shift, const unsigned mask) const {
  return (word >> shift) & mask;
}

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi) { return o << "CSC Shower: " << digi.bits(); }
