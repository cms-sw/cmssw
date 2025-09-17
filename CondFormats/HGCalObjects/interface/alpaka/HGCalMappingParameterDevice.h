#ifndef CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDevice_h
#define CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterSoA.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcal {

    using HGCalMappingModuleParamDevice = PortableCollection<::hgcal::HGCalMappingModuleParamSoA>;
    using HGCalMappingModuleTriggerParamDevice = PortableCollection<::hgcal::HGCalMappingModuleTriggerParamSoA>;
    using HGCalMappingCellParamDevice = PortableCollection<::hgcal::HGCalMappingCellParamSoA>;
    using HGCalDenseIndexInfoDevice = PortableCollection<::hgcal::HGCalDenseIndexInfoSoA>;
    using HGCalDenseIndexTriggerInfoDevice = PortableCollection<::hgcal::HGCalDenseIndexTriggerInfoSoA>;

    using HGCalMappingModuleParamHost = ::hgcal::HGCalMappingModuleParamHost;
    using HGCalMappingModuleTriggerParamHost = ::hgcal::HGCalMappingModuleTriggerParamHost;
    using HGCalMappingCellParamHost = ::hgcal::HGCalMappingCellParamHost;
    using HGCalDenseIndexInfoHost = ::hgcal::HGCalDenseIndexInfoHost;
    using HGCalDenseIndexTriggerInfoHost = ::hgcal::HGCalDenseIndexTriggerInfoHost;

  }  // namespace hgcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_HGCalMappingParameterDevice_h
