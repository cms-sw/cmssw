#ifndef HeterogeneousCore_Product_HeterogeneousDeviceId_h
#define HeterogeneousCore_Product_HeterogeneousDeviceId_h

/**
 * Enumerator for heterogeneous device types
 */
enum class HeterogeneousDevice {
  kCPU = 0,
  kGPUMock,
  kGPUCuda,
  kSize
};

namespace heterogeneous {
  template <HeterogeneousDevice Device>
  struct HeterogeneousDeviceTag {
    constexpr static HeterogeneousDevice value = Device;
  };
}

/**
 * Class to represent an identifier for a heterogeneous device.
 * Contains device type and an integer identifier.
 */
class HeterogeneousDeviceId {
public:
  constexpr static auto kInvalidDevice = HeterogeneousDevice::kSize;

  HeterogeneousDeviceId():
    deviceType_(kInvalidDevice),
    deviceId_(0)
  {}
  explicit HeterogeneousDeviceId(HeterogeneousDevice device, unsigned int id=0):
    deviceType_(device), deviceId_(id)
  {}

  HeterogeneousDevice deviceType() const { return deviceType_; }

  unsigned int deviceId() const { return deviceId_; }
private:
  HeterogeneousDevice deviceType_;
  unsigned int deviceId_;
};

#endif
