#ifndef FWCore_Concurrency_numaLocality_h
#define FWCore_Concurrency_numaLocality_h
//
// The memory-policy part of process.options.numaLocality: allocate the memory on the NUMA node of the cpu that first
// touches it, as `numactl --localalloc` does. An explicit policy also takes the job out of the kernel's automatic NUMA
// balancing, whose page scans and migrations can cost more than they gain (as measured for the HLT).
// The rest of numaLocality (per-domain arenas, placement of the per-stream objects, EventSetup replication) is
// described in StreamNumaDomains.h.
//

namespace edm::numa {

  // true if the kernel lets the job use the NUMA functions (libnuma's numa_available(); for example a container whose
  // seccomp filter forbids get_mempolicy and set_mempolicy does not)
  bool numaAvailable();

  // Set a local memory policy for the calling thread and the threads it creates afterwards. Called by cmsRun with
  // numaLocality, before the TBB worker threads are created, so that they inherit it. Returns true if the policy was
  // set.
  bool setLocalMemoryPolicy();

  // true if setLocalMemoryPolicy() has set the policy
  bool localMemoryPolicy();

}  // namespace edm::numa

#endif
