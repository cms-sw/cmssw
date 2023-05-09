#ifndef HeterogeneousCore_ROCmServices_interface_ROCmInterface_h
#define HeterogeneousCore_ROCmServices_interface_ROCmInterface_h

#include <utility>

class ROCmInterface {
public:
  ROCmInterface() = default;
  virtual ~ROCmInterface() = default;

  virtual bool enabled() const = 0;

  virtual int numberOfDevices() const = 0;

  // Returns the (major, minor) compute capability of the given device.
  virtual std::pair<int, int> computeCapability(int device) const = 0;
};

#endif  // HeterogeneousCore_ROCmServices_interface_ROCmInterface_h
