#ifndef CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDevice_h
#define CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterSoA.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    using HGCalMappingModuleParamDevice = PortableCollection<::hgcal::HGCalMappingModuleParamSoA>;
    using HGCalMappingCellParamDevice = PortableCollection<::hgcal::HGCalMappingCellParamSoA>;
    using HGCalDenseIndexInfoDevice = PortableCollection<::hgcal::HGCalDenseIndexInfoSoA>;

    using HGCalMappingModuleParamHost = ::hgcal::HGCalMappingModuleParamHost;
    using HGCalMappingCellParamHost = ::hgcal::HGCalMappingCellParamHost;
    using HGCalDenseIndexInfoHost = ::hgcal::HGCalDenseIndexInfoHost;

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDevice_h
