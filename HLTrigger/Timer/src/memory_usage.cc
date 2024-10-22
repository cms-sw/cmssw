#include <iostream>
#include <dlfcn.h>

#include "HLTrigger/Timer/interface/memory_usage.h"

// see <jemalloc/jemalloc.h>
extern "C" {
typedef int (*mallctl_t)(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
typedef int (*mallctlnametomib_t)(const char *name, size_t *mibp, size_t *miblenp);
typedef int (*mallctlbymib_t)(const size_t *mib, size_t miblen, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
}

namespace {
  bool initialise();
  bool initialise_peak();
  const uint64_t *initialise_thread_allocated_p();
  const uint64_t *initialise_thread_deallocated_p();

  const uint64_t zero = 0UL;
  size_t mib_peak_read[3];   // management information base for "thread.peak.read"
  size_t mib_peak_reset[3];  // management information base for "thread.peak.reset"
  thread_local const uint64_t *thread_allocated_p = initialise_thread_allocated_p();
  thread_local const uint64_t *thread_deallocated_p = initialise_thread_deallocated_p();

  mallctl_t mallctl = nullptr;
  mallctlnametomib_t mallctlnametomib = nullptr;
  mallctlbymib_t mallctlbymib = nullptr;
  const bool have_jemalloc_and_stats = initialise();
  const bool have_jemalloc_and_peak = have_jemalloc_and_stats and initialise_peak();

  bool initialise() {
    // check if mallctl and friends are available, if we are using jemalloc
    mallctl = (mallctl_t)::dlsym(RTLD_DEFAULT, "mallctl");
    if (mallctl == nullptr)
      return false;
    mallctlnametomib = (mallctlnametomib_t)::dlsym(RTLD_DEFAULT, "mallctlnametomib");
    if (mallctlnametomib == nullptr)
      return false;
    mallctlbymib = (mallctlbymib_t)::dlsym(RTLD_DEFAULT, "mallctlbymib");
    if (mallctlbymib == nullptr)
      return false;

    // check if the statistics are available, if --enable-stats was specified at build time
    bool enable_stats = false;
    size_t bool_s = sizeof(bool);
    mallctl("config.stats", &enable_stats, &bool_s, nullptr, 0);
    return enable_stats;
  }

  bool initialise_peak() {
    // check if thread.peak.read and thread.peak.reset are available
    size_t miblen = 3;
    if (mallctlnametomib("thread.peak.read", mib_peak_read, &miblen) != 0)
      return false;
    if (mallctlnametomib("thread.peak.reset", mib_peak_reset, &miblen) != 0)
      return false;
    return true;
  }

  const uint64_t *initialise_thread_allocated_p() {
    const uint64_t *stats = &zero;
    size_t ptr_s = sizeof(uint64_t *);

    if (have_jemalloc_and_stats)
      // get a pointer to the thread-specific allocation statistics
      mallctl("thread.allocatedp", &stats, &ptr_s, nullptr, 0);

    return stats;
  }

  const uint64_t *initialise_thread_deallocated_p() {
    const uint64_t *stats = &zero;
    size_t ptr_s = sizeof(uint64_t *);

    if (have_jemalloc_and_stats)
      // get a pointer to the thread-specific allocation statistics
      mallctl("thread.deallocatedp", &stats, &ptr_s, nullptr, 0);

    return stats;
  }

}  // namespace

bool memory_usage::is_available() { return have_jemalloc_and_stats; }

uint64_t memory_usage::allocated() { return *thread_allocated_p; }

uint64_t memory_usage::deallocated() { return *thread_deallocated_p; }

uint64_t memory_usage::peak() {
  uint64_t peak = 0;
  size_t size = sizeof(uint64_t);
  if (have_jemalloc_and_peak)
    mallctlbymib(mib_peak_read, 3, &peak, &size, nullptr, 0);
  return peak;
}

void memory_usage::reset_peak() {
  if (have_jemalloc_and_peak)
    mallctlbymib(mib_peak_reset, 3, nullptr, nullptr, nullptr, 0);
}
