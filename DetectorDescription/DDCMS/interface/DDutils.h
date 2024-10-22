#ifndef DETECTOR_DESCRIPTION_DDCMS_DDUTILS_H
#define DETECTOR_DESCRIPTION_DDCMS_DDUTILS_H
#include "DD4hep/DD4hepUnits.h"

namespace cms {
  template <class NumType>
  inline constexpr NumType convert2mm(NumType length) {
    return (length / dd4hep::mm);
  }
}  // namespace cms
#endif
