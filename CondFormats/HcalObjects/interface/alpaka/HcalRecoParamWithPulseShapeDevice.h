#ifndef CondFormats_HcalObjects_interface_alpaka_HcalRecoParamWithPulseShapeDevice_h
#define CondFormats_HcalObjects_interface_alpaka_HcalRecoParamWithPulseShapeDevice_h

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeT.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace hcal {
    using HcalRecoParamWithPulseShapeDevice = ::hcal::HcalRecoParamWithPulseShapeT<Device>;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
