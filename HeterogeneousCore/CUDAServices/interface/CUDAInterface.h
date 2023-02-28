#ifndef HeterogeneousCore_CUDAServices_interface_CUDAInterface
#define HeterogeneousCore_CUDAServices_interface_CUDAInterface

#include <utility>

class CUDAInterface {
public:
  CUDAInterface() = default;
  virtual ~CUDAInterface() = default;

  virtual bool enabled() const = 0;

  virtual int numberOfDevices() const = 0;

  // Returns the (major, minor) CUDA compute capability of the given device.
  virtual std::pair<int, int> computeCapability(int device) const = 0;
};

#endif  // HeterogeneousCore_CUDAServices_interface_CUDAInterface
