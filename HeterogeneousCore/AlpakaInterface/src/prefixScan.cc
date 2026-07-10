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
        << "Shared memory limit exceeded for prefix scan of " << nElements << " elements in " << nBlocks
        << " blocks. Required shared memory: " << requiredSharedMem << " bytes. Shared memory limit: " << sharedMemLimit
        << " bytes.";
  }

  void throwIterativePrefixScanMaxLevelsExceeded(const size_t nElements, const uint32_t nLevels) {
    throw cms::Exception("IterativePrefixScanMaxLevelsExceeded")
        << "Requested an iterative prefix scan for " << nElements << " elements.\n"
        << "The problem was split into " << nLevels
        << " levels, which exceeds the maximum supported number of levels of " << iterativePrefixScanMaxLevels
        << " (enough for " << iterativePrefixScanThreads << "^" << iterativePrefixScanMaxLevels << " elements).\n"
        << "Consider increasing the value of iterativePrefixScanMaxLevels or reducing the problem's size.";
  }

}  // namespace cms::alpakatools
