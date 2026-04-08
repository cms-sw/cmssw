#include <alpaka/alpaka.hpp>
#include <cstdint>

#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"

namespace cms::alpakatools {
  void throwSharedMemoryLimitExceeded(const size_t nElements,
                                      const uint32_t nBlocks,
                                      const size_t requiredSharedMem,
                                      const size_t sharedMemLimit) {
    throw cms::Exception("SharedMemoryLimitExceeded")
        << "OneToManyAssoc: Shared memory limit exceeded for prefix scan of " << nElements << " elements in " << nBlocks
        << " blocks. Required shared memory: " << requiredSharedMem << " bytes. Shared memory limit: " << sharedMemLimit
        << " bytes.";
  }
}  // namespace cms::alpakatools
