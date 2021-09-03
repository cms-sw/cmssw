#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableCCLUT.h"

CSCL1TPLookupTableCCLUT::CSCL1TPLookupTableCCLUT() {
  cclutPosition_.reserve(5);
  cclutSlope_.reserve(5);
}

void CSCL1TPLookupTableCCLUT::set_cclutPosition(t_lut lut) { cclutPosition_ = std::move(lut); }

void CSCL1TPLookupTableCCLUT::set_cclutSlope(t_lut lut) { cclutSlope_ = std::move(lut); }

unsigned CSCL1TPLookupTableCCLUT::cclutPosition(unsigned pattern, unsigned code) const {
  return cclutPosition_.at(pattern)[code];
}

unsigned CSCL1TPLookupTableCCLUT::cclutSlope(unsigned pattern, unsigned code) const {
  return cclutSlope_.at(pattern)[code];
}
