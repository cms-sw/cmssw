#ifndef memory_allocation_stats_h
#define memory_allocation_stats_h

#include <cstdint>
#include <tuple>

#include <dlfcn.h>

extern "C" {
  typedef
  int (*mallctl_t)(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
}

namespace cms {

namespace detail {

class MemoryAllocationMonitorTLS {
public:
  MemoryAllocationMonitorTLS() {

    // check if mallctl is available, if we are using jemalloc.
    mallctl_t mallctl = (mallctl_t) ::dlsym(RTLD_DEFAULT, "mallctl");
    if (mallctl == nullptr) {
      static const uint64_t zero = 0LL;
      thread_allocated_p   = & zero;
      thread_deallocated_p = & zero;
      return;
    }

    // check if the statistics are available, if --enable-stats was specified at build time.
    bool enable_stats = false;
    size_t bool_s = sizeof(bool);
    mallctl("config.stats", & enable_stats, & bool_s, nullptr, 0);
    if (not enable_stats) {
      static const uint64_t zero = 0LL;
      thread_allocated_p   = & zero;
      thread_deallocated_p = & zero;
      return;
    }

    // get pointers to the thread-specific allocation statistics.
    size_t ptr_s = sizeof(uint64_t *);
    mallctl("thread.allocatedp",   & thread_allocated_p,   & ptr_s, nullptr, 0);
    mallctl("thread.deallocatedp", & thread_deallocated_p, & ptr_s, nullptr, 0);
  }

  std::tuple<uint64_t,uint64_t> read() const {
    return std::make_tuple(* thread_allocated_p, * thread_deallocated_p);
  }

private:
  const uint64_t * thread_allocated_p;
  const uint64_t * thread_deallocated_p;
};

}

class MemoryAllocationMonitor {
public:
  static std::tuple<uint64_t,uint64_t> read() {
    return tls.read();
  }

private:
  static thread_local detail::MemoryAllocationMonitorTLS tls;
};

}

#endif // memory_allocation_stats_h
