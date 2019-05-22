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

#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"

using namespace pos;

PixelMaskBase::PixelMaskBase(std::string description, std::string creator, std::string date)
    : PixelConfigBase(description, creator, date) {}

PixelMaskBase::~PixelMaskBase() {}

void PixelMaskBase::setOverride(PixelMaskOverrideBase* override) { maskOverride_ = override; }

std::ostream& operator<<(std::ostream& s, const PixelMaskBase& mask) {
  s << mask.getMaskBits(0) << std::endl;

  return s;
}
