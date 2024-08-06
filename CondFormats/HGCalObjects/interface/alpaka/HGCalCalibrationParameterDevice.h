#ifndef CondFormats_HGCalObjects_interface_alpaka_HGCalCalibrationParameterDevice_h
#define CondFormats_HGCalObjects_interface_alpaka_HGCalCalibrationParameterDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterSoA.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    using namespace ::hgcalrechit;
    using HGCalCalibParamDevice = PortableCollection<HGCalCalibParamSoA>;
    //using HGCalChannelConfigParamDevice = PortableCollection<HGCalChannelConfigParamSoA>;
    using HGCalConfigParamDevice = PortableCollection<HGCalConfigParamSoA>;

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_HGCalObjects_interface_alpaka_HGCalCalibrationParameterDevice_h
