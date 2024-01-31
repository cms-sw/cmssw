#ifndef RecoLocalTracker_SiPixelRecHits_interface_PixelCPEFastParamsDevice_h
#define RecoLocalTracker_SiPixelRecHits_interface_PixelCPEFastParamsDevice_h

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

template <typename TDev, typename TrackerTraits>
class PixelCPEFastParamsDevice {
public:
  using Buffer = cms::alpakatools::device_buffer<TDev, pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;
  using ConstBuffer = cms::alpakatools::const_device_buffer<TDev, pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>;

  template <typename TQueue>
  PixelCPEFastParamsDevice(TQueue queue)
      : buffer_(cms::alpakatools::make_device_buffer<pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>>(queue)) {}

  // non-copyable
  PixelCPEFastParamsDevice(PixelCPEFastParamsDevice const&) = delete;
  PixelCPEFastParamsDevice& operator=(PixelCPEFastParamsDevice const&) = delete;

  // movable
  PixelCPEFastParamsDevice(PixelCPEFastParamsDevice&&) = default;
  PixelCPEFastParamsDevice& operator=(PixelCPEFastParamsDevice&&) = default;

  // default destructor
  ~PixelCPEFastParamsDevice() = default;

  // access the buffer
  Buffer buffer() { return buffer_; }
  ConstBuffer buffer() const { return buffer_; }
  ConstBuffer const_buffer() const { return buffer_; }

  auto size() const { return alpaka::getExtentProduct(buffer_); }

  pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits> const* data() const { return buffer_.data(); }

private:
  Buffer buffer_;
};

#endif  // RecoLocalTracker_SiPixelRecHits_interface_PixelCPEFastParamsDevice_h
