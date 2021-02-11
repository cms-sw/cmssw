#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <iostream>

using namespace std;

/// Constructors
CSCShowerDigi::CSCShowerDigi(const uint8_t bits, const uint16_t cscID) : bits_(bits), cscID_(cscID) {}

/// Default
CSCShowerDigi::CSCShowerDigi() : bits_(0), cscID_(0) {}

bool CSCShowerDigi::isValid() const {
  return isLoose();
}

bool CSCShowerDigi::isLoose() const { return bits_ >= kLoose; }

bool CSCShowerDigi::isNominal() const { return bits_ >= kNominal; }

bool CSCShowerDigi::isTight() const { return bits_ >= kTight; }

std::ostream& operator<<(std::ostream& o, const CSCShowerDigi& digi) { return o << "CSC Shower: " << digi.bits(); }
