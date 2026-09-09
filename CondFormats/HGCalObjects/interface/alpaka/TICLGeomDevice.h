#ifndef CondFormats_HGCalObjects_interface_alpaka_TICLGeomDevice_h
#define CondFormats_HGCalObjects_interface_alpaka_TICLGeomDevice_h

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::TICLGeomHost;
  using TICLGeomDevice = PortableCollection<TICLGeomSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_TICLGeomDevice_h
