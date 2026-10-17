#ifndef CondFormats_HGCalObjects_interface_HGCalAgeingCondsHost_h
#define CondFormats_HGCalObjects_interface_HGCalAgeingCondsHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsSoA.h"

namespace hgcal {

  //SoA with detailed indices corresponding to the dense index in use
  using HGCalAgeingCondsHost = PortableHostCollection<HGCalAgeingCondsSoA>;

}  // namespace hgcal

#endif  // CondFormats_HGCalObjects_interface_HGCalAgeingCondsHost_h
