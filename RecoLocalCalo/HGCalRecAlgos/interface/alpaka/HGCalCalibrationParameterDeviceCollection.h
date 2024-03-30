#ifndef RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCalibrationParameterDeviceCollection_h
#define RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCalibrationParameterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCalibrationParameterSoA.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCalibrationParameterHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    using namespace ::hgcalrechit;
    using HGCalCalibParamDeviceCollection = PortableCollection<HGCalCalibParamSoA>;
    //using HGCalChannelConfigParamDeviceCollection = PortableCollection<HGCalChannelConfigParamSoA>;
    using HGCalConfigParamDeviceCollection = PortableCollection<HGCalConfigParamSoA>;

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_alpaka_HGCalCalibrationParameterDeviceCollection_h
