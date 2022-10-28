#ifndef HeterogeneousCore_CUDAUtilities_interface_cachingAllocators_h
#define HeterogeneousCore_CUDAUtilities_interface_cachingAllocators_h

namespace cms::cuda::allocator {
  // Use caching or not
  constexpr bool useCaching = true;

  // these intended to be called only from CUDAService
  void cachingAllocatorsConstruct();
  void cachingAllocatorsFreeCached();
}  // namespace cms::cuda::allocator

#endif
