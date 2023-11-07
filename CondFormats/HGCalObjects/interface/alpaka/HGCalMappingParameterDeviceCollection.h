#ifndef CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDeviceCollection_h
#define CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterSoA.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    using namespace ::hgcal;
    using HGCalMappingModuleParamDeviceCollection = PortableCollection<HGCalMappingModuleParamSoA>;
    using HGCalMappingCellParamDeviceCollection = PortableCollection<HGCalMappingCellParamSoA>;

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDeviceCollection_h
