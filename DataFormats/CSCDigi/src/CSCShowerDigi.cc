#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

using namespace std;

/// Constructors
CSCShowerDigi::CSCShowerDigi(const uint16_t bitsInTime, const uint16_t bitsOutOfTime, const uint16_t cscID)
    : bitsInTime_(bitsInTime), bitsOutOfTime_(bitsOutOfTime), cscID_(cscID) {}

/// Default
CSCShowerDigi::CSCShowerDigi() : bitsInTime_(0), bitsOutOfTime_(0), cscID_(0) {}

void CSCShowerDigi::clear() {
  bitsInTime_ = 0;
  bitsOutOfTime_ = 0;
  cscID_ = 0;
}

bool CSCShowerDigi::isValid() const {
  // any loose shower is valid
  return isLooseInTime() or isLooseOutOfTime();
}

bool CSCShowerDigi::isLooseInTime() const { return bitsInTime() >= kLoose; }

bool CSCShowerDigi::isNominalInTime() const { return bitsInTime() >= kNominal; }

bool CSCShowerDigi::isTightInTime() const { return bitsInTime() >= kTight; }

bool CSCShowerDigi::isLooseOutOfTime() const { return bitsOutOfTime() >= kLoose; }

bool CSCShowerDigi::isNominalOutOfTime() const { return bitsOutOfTime() >= kNominal; }

bool CSCShowerDigi::isTightOutOfTime() const { return bitsOutOfTime() >= kTight; }

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi) {
  return o << "CSC Shower: in-time bits " << digi.bitsInTime() << ", out-of-time bits " << digi.bitsOutOfTime();
}
