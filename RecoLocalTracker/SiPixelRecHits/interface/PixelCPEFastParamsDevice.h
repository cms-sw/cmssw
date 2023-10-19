#ifndef DataFormats_PixelCPEFastParams_interface_PixelCPEFastParamsDevice_h
#define DataFormats_PixelCPEFastParams_interface_PixelCPEFastParamsDevice_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGenericBase.h"

#include "pixelCPEforDevice.h"

template <typename TDev, typename TrackerTraits>
class PixelCPEFastParamsDevice {
public:
  using Buffer = cms::alpakatools::device_buffer<TDev, pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;

  template <typename TQueue>
  PixelCPEFastParamsDevice(TQueue queue)
      : buffer_(cms::alpakatools::make_device_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>(queue)) {}

  Buffer buffer() { return buffer_; }
  ConstBuffer buffer() const { return buffer_; }
  ConstBuffer const_buffer() const { return buffer_; }
  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* data() const { return buffer_.data(); }
  auto size() const { return alpaka::getExtentProduct(buffer_); }

private:
  Buffer buffer_;
};

#endif
