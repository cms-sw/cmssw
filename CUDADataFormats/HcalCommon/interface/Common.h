#ifndef CUDADataFormats_HcalCommon_interface_Common_h
#define CUDADataFormats_HcalCommon_interface_Common_h

#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

namespace hcal {
  namespace common {

    // FIXME: not able to get enums to work with genreflex
    namespace tags {

      struct Vec {};
      struct Ptr {};

    }  // namespace tags

    template <typename tag>
    struct AddSize {};

    template <>
    struct AddSize<tags::Ptr> {
      uint32_t size;
    };

    struct ViewStoragePolicy {
      using TagType = tags::Ptr;

      template <typename T>
      struct StorageSelector {
        using type = T*;
      };
    };

    template <template <typename> typename Allocator = std::allocator>
    struct VecStoragePolicy {
      using TagType = tags::Vec;

      template <typename T>
      struct StorageSelector {
        using type = std::vector<T, Allocator<T>>;
      };
    };

  }  // namespace common
}  // namespace hcal

// FIXME: move into common namespace
namespace hcal {

  template <typename T>
  using CUDAHostAllocatorAlias = cms::cuda::HostAllocator<T>;

}

#endif  // CUDADataFormats_HcalCommon_interface_Common_h
