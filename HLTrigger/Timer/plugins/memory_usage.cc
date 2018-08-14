#include <iostream>
#include <dlfcn.h>

#include "memory_usage.h"

// see <jemalloc/jemalloc.h>
extern "C" {
  typedef
  int (*mallctl_t)(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
}

namespace {
  bool initialise();
  const uint64_t * initialise_thread_allocated_p();
  const uint64_t * initialise_thread_deallocated_p();

  const uint64_t zero = 0UL;
  thread_local const uint64_t * thread_allocated_p   = initialise_thread_allocated_p();
  thread_local const uint64_t * thread_deallocated_p = initialise_thread_deallocated_p();

  mallctl_t mallctl = nullptr;
  const bool have_jemalloc_and_stats = initialise();


  bool initialise()
  {
    // check if mallctl is available, if we are using jemalloc
    mallctl = (mallctl_t) ::dlsym(RTLD_DEFAULT, "mallctl");
    if (mallctl == nullptr) 
      return false;

    // check if the statistics are available, if --enable-stats was specified at build time
    bool enable_stats = false;
    size_t bool_s = sizeof(bool);
    mallctl("config.stats", & enable_stats, & bool_s, nullptr, 0);
    return enable_stats;
  }

  const uint64_t * initialise_thread_allocated_p()
  {
    const uint64_t * stats = & zero;
    size_t ptr_s = sizeof(uint64_t *);

    if (have_jemalloc_and_stats)
      // get pointers to the thread-specific allocation statistics
      mallctl("thread.allocatedp", & stats, & ptr_s, nullptr, 0);

    return stats;
  }

  const uint64_t * initialise_thread_deallocated_p()
  {
    const uint64_t * stats = & zero;
    size_t ptr_s = sizeof(uint64_t *);

    if (have_jemalloc_and_stats)
      // get pointers to the thread-specific allocation statistics
      mallctl("thread.deallocatedp", & stats, & ptr_s, nullptr, 0);

    return stats;
  }

} // namespace

bool memory_usage::is_available()
{
  return have_jemalloc_and_stats;
}

uint64_t memory_usage::allocated()
{
  return * thread_allocated_p;
}

uint64_t memory_usage::deallocated()
{
  return * thread_deallocated_p;
}

