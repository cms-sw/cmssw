//
// This class provide a base class for the
// pixel mask data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// All applications should just use this
// interface and not care about the specific
// implementation
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include <vector>
#include <iostream>

using namespace pos;

PixelTrimBase::PixelTrimBase(std::string description, std::string creator, std::string date)
    : PixelConfigBase(description, creator, date) {}

PixelTrimBase::~PixelTrimBase() {}

void PixelTrimBase::setOverride(PixelTrimOverrideBase* override) { trimOverride_ = override; }

std::ostream& operator<<(std::ostream& s, const PixelTrimBase& trim) {
  s << trim.getTrimBits(0) << std::endl;

  return s;
}
