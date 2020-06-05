#include "DataFormats/Common/interface/Wrapper.h"
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"

namespace hcal {

  // explicit template instantiations
  template struct RecHitCollection<common::ViewStoragePolicy>;

  template struct RecHitCollection<common::VecStoragePolicy<std::allocator>>;

  template struct RecHitCollection<common::VecStoragePolicy<CUDAHostAllocatorAlias>>;

}  // namespace hcal
