#ifndef Geometry_HGCalCommonData_HGCalDD4HepHelper_h
#define Geometry_HGCalCommonData_HGCalDD4HepHelper_h
#include "DD4hep/DD4hepUnits.h"

namespace HGCalDD4HepHelper {
  template <class NumType>
  inline constexpr NumType convert2mm(NumType length) {
    return (length / dd4hep::mm);
  }
}  // namespace HGCalDD4HepHelper
#endif
