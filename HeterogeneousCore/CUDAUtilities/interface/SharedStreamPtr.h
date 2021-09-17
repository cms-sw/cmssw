#ifndef HeterogeneousCore_CUDAUtilities_SharedStreamPtr_h
#define HeterogeneousCore_CUDAUtilities_SharedStreamPtr_h

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    // cudaStream_t itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
    using SharedStreamPtr = std::shared_ptr<std::remove_pointer_t<cudaStream_t>>;
  }  // namespace cuda
}  // namespace cms

#endif
