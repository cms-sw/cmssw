#include "FWCore/Concurrency/interface/numaLocality.h"

#include <numa.h>

namespace {
  // set by edm::numa::setLocalMemoryPolicy()
  bool localPolicy = false;
}  // namespace

namespace edm::numa {

  bool numaAvailable() { return numa_available() >= 0; }

  bool setLocalMemoryPolicy() {
    if (not numaAvailable()) {
      return false;
    }
    // also if all the job's cpus are on one node: the memory goes where the default policy would put it, but the kernel
    // stops scanning the job's pages for automatic NUMA balancing
    numa_set_localalloc();
    localPolicy = true;
    return true;
  }

  bool localMemoryPolicy() { return localPolicy; }

}  // namespace edm::numa
