#ifndef CondFormats_HGCalObjects_interface_alpaka_TICLGeomLookupDevice_h
#define CondFormats_HGCalObjects_interface_alpaka_TICLGeomLookupDevice_h

#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using ::TICLGeomLookupHost;
  using TICLGeomLookupDevice = PortableCollection<TICLGeomLookupSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_TICLGeomLookupDevice_h
