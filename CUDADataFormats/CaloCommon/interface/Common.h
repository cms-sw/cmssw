#ifndef CUDADataFormats_CaloCommon_interface_Common_h
#define CUDADataFormats_CaloCommon_interface_Common_h

#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

namespace calo {
  namespace common {

    // FIXME: not able to get enums to work with genreflex
    namespace tags {

      struct Vec {};
      struct Ptr {};
      struct DevPtr {};

    }  // namespace tags

    template <typename tag>
    struct AddSize {};

    template <>
    struct AddSize<tags::Ptr> {
      uint32_t size;
    };

    template <>
    struct AddSize<tags::DevPtr> {
      uint32_t size;
    };

    struct ViewStoragePolicy {
      using TagType = tags::Ptr;

      template <typename T>
      struct StorageSelector {
        using type = T*;
      };
    };

    struct DevStoragePolicy {
      using TagType = tags::DevPtr;

      template <typename T>
      struct StorageSelector {
        using type = cms::cuda::device::unique_ptr<T[]>;
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

    template <typename T>
    using CUDAHostAllocatorAlias = cms::cuda::HostAllocator<T>;

  }  // namespace common
}  // namespace calo

#endif  // CUDADataFormats_CaloCommon_interface_Common_h
