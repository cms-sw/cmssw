#ifndef HeterogeneousCore_Product_HeterogeneousProductBase_h
#define HeterogeneousCore_Product_HeterogeneousProductBase_h

#include "HeterogeneousCore/Product/interface/HeterogeneousDeviceId.h"

#include <array>
#include <bitset>
#include <mutex>

namespace heterogeneous {
  constexpr const unsigned int kMaxDevices = 16;
  using DeviceBitSet = std::bitset<kMaxDevices>;
}

// For type erasure to ease dictionary generation
class HeterogeneousProductBase {
public:
  // TODO: Given we'll likely have the data on one or at most a couple
  // of devices, storing the information in a "dense" bit pattern may
  // be overkill. Maybe a "sparse" presentation would be sufficient
  // and easier to deal with?
  using BitSet = heterogeneous::DeviceBitSet;
  using BitSetArray = std::array<BitSet, static_cast<unsigned int>(HeterogeneousDevice::kSize)>;

  virtual ~HeterogeneousProductBase() = 0;

  bool isProductOn(HeterogeneousDevice loc) const {
    // should this be protected with the mutex?
    return location_[static_cast<unsigned int>(loc)].any();
  }
  BitSet onDevices(HeterogeneousDevice loc) const {
    // should this be protected with the mutex?
    return location_[static_cast<unsigned int>(loc)];
  }

protected:
  mutable std::mutex mutex_;
  mutable BitSetArray location_;
};

#endif
