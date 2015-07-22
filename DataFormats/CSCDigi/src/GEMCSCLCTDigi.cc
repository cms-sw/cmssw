#include "DataFormats/CSCDigi/interface/GEMCSCLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

/// Constructors
GEMCSCLCTDigi::GEMCSCLCTDigi(const CSCCorrelatedLCTDigi digi, float bend) : 
  digi_(digi),
  bend_(bend)
{}

/// Default
GEMCSCLCTDigi::GEMCSCLCTDigi() {
}

/// Comparison
bool GEMCSCLCTDigi::operator==(const GEMCSCLCTDigi &rhs) const {
  return ( digi_ == rhs.getDigi() && bend_ == rhs.getBend() );
}
