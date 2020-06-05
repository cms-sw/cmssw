#include "DataFormats/Common/interface/Wrapper.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"

namespace hcal {

  // instantiate what we know will be used
  template struct DigiCollection<Flavor01, common::ViewStoragePolicy>;

  template struct DigiCollection<Flavor2, common::ViewStoragePolicy>;

  template struct DigiCollection<Flavor3, common::ViewStoragePolicy>;

  template struct DigiCollection<Flavor4, common::ViewStoragePolicy>;

  template struct DigiCollection<Flavor5, common::ViewStoragePolicy>;

  template struct DigiCollection<Flavor01, common::VecStoragePolicy<std::allocator>>;

  template struct DigiCollection<Flavor2, common::VecStoragePolicy<std::allocator>>;

  template struct DigiCollection<Flavor3, common::VecStoragePolicy<std::allocator>>;

  template struct DigiCollection<Flavor4, common::VecStoragePolicy<std::allocator>>;

  template struct DigiCollection<Flavor5, common::VecStoragePolicy<std::allocator>>;

  template struct DigiCollection<Flavor01, common::VecStoragePolicy<CUDAHostAllocatorAlias>>;

  template struct DigiCollection<Flavor2, common::VecStoragePolicy<CUDAHostAllocatorAlias>>;

  template struct DigiCollection<Flavor3, common::VecStoragePolicy<CUDAHostAllocatorAlias>>;

  template struct DigiCollection<Flavor4, common::VecStoragePolicy<CUDAHostAllocatorAlias>>;

  template struct DigiCollection<Flavor5, common::VecStoragePolicy<CUDAHostAllocatorAlias>>;

}  // namespace hcal
