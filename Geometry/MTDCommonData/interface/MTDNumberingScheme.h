#ifndef MTDNumberingScheme_h
#define MTDNumberingScheme_h

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdint>

class MTDNumberingScheme {
 public:
  MTDNumberingScheme();
  virtual ~MTDNumberingScheme();
  virtual uint32_t getUnitID(const MTDBaseNumber& baseNumber) const = 0;
};

#endif
