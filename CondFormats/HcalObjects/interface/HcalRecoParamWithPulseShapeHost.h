#ifndef CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHost_h
#define CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHost_h

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeT.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace hcal {
  using HcalRecoParamWithPulseShapeHost = HcalRecoParamWithPulseShapeT<alpaka::DevCpu>;
}
namespace cms::alpakatools {
  template <>
  struct CopyToDevice<::hcal::HcalRecoParamWithPulseShapeHost> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::hcal::HcalRecoParamWithPulseShapeHost const& hostProduct) {
      using RecoParamCopy = CopyToDevice<::hcal::HcalRecoParamWithPulseShapeHost::RecoParamCollection>;
      using PulseShapeCopy = CopyToDevice<::hcal::HcalRecoParamWithPulseShapeHost::PulseShapeCollection>;
      using Device = alpaka::Dev<TQueue>;
      return ::hcal::HcalRecoParamWithPulseShapeT<Device>(RecoParamCopy::copyAsync(queue, hostProduct.recoParam()),
                                                          PulseShapeCopy::copyAsync(queue, hostProduct.pulseShape()));
    }
  };
}  // namespace cms::alpakatools

#endif
