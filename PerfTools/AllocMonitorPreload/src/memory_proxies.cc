#include <memory>
#include <cassert>
#include <atomic>
#include <array>
#include <mutex>
#include <limits>
#include <cstddef>
#include <malloc.h>
#define ALLOC_USE_PTHREADS
#if defined(ALLOC_USE_PTHREADS)
#include <pthread.h>
#else
#include <unistd.h>
#include <sys/syscall.h>
#endif

#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <dlfcn.h>  // dlsym

#if !defined(__x86_64__) && !defined(__i386__)
#define USE_LOCAL_MALLOC
#endif
#if defined(__GLIBC__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ < 28)
//needed for sl7
#define USE_LOCAL_MALLOC
#endif

namespace {
  std::atomic<bool>& alloc_monitor_running_state() {
    static std::atomic<bool> s_state = false;
    return s_state;
  }

  template <typename T>
  T get(const char* iName) {
    void* original = dlsym(RTLD_NEXT, iName);
    assert(original);
    return reinterpret_cast<T>(original);
  }

  inline auto thread_id() {
#if defined(ALLOC_USE_PTHREADS)
    /*NOTE: if use pthread_self, the values returned by linux had                                                                                                                                                                                      lots of hash collisions when using a simple % hash. Worked                                                                                                                                                                                        better if first divided value by 0x700 and then did %.                                                                                                                                                                                            [test done on el8] */
    return pthread_self();
#else
    return syscall(SYS_gettid);
#endif
  }
#ifdef USE_LOCAL_MALLOC
  // this is a very simple-minded allocator used for any allocations
  // before we've finished our setup.  In particular, this avoids a
  // chicken/egg problem if dlsym() allocates any memory.
  // Size was chosen to be 2x what ARM64 uses as an emergency buffer
  // for libstdc++ exception handling.
  constexpr auto max_align = alignof(std::max_align_t);
  alignas(max_align) char tmpbuff[131072];
  unsigned long tmppos = 0;
  unsigned long tmpallocs = 0;

  void* local_malloc(size_t size) noexcept {
    // round up so next alloc is aligned
    size = ((size + max_align - 1) / max_align) * max_align;
    if (tmppos + size < sizeof(tmpbuff)) {
      void* retptr = tmpbuff + tmppos;
      tmppos += size;
      ++tmpallocs;
      return retptr;
    } else {
      return nullptr;
    }
  }

  //can use local_malloc since static memory buffers are guaranteed to be zero initialized
  void* local_calloc(size_t nitems, size_t item_size) noexcept { return local_malloc(nitems * item_size); }

  inline bool is_local_alloc(void* ptr) noexcept { return ptr >= (void*)tmpbuff && ptr <= (void*)(tmpbuff + tmppos); }

  // the pointers in this struct should only be modified during
  // global construction at program startup, so thread safety
  // should not be an issue.
  struct Originals {
    inline static void init() noexcept {
      if (not set) {
        set = true;  // must be first to avoid recursion
        malloc = get<decltype(&::malloc)>("malloc");
        calloc = get<decltype(&::calloc)>("calloc");
      }
    }
    CMS_SA_ALLOW static decltype(&::malloc) malloc;
    CMS_SA_ALLOW static decltype(&::calloc) calloc;
    CMS_SA_ALLOW static bool set;
  };

  decltype(&::malloc) Originals::malloc = local_malloc;
  decltype(&::calloc) Originals::calloc = local_calloc;
  bool Originals::set = false;
#else
  constexpr inline bool is_local_alloc(void* ptr) noexcept { return false; }
#endif

  struct ThreadTracker {
    static constexpr unsigned int kEntries = 128;
    using entry_type = decltype(thread_id());
    std::array<std::atomic<entry_type>, kEntries> used_threads_;
    std::array<std::mutex, kEntries> used_threads_mutex_;

    ThreadTracker() {
      //put a value which will not match the % used when looking up the entry
      entry_type entry = 0;
      for (auto& v : used_threads_) {
        v = ++entry;
      }
    }

    std::size_t thread_index(entry_type id) const {
#if defined(ALLOC_USE_PTHREADS)
      return (id / 0x700) % kEntries;
#else
      return id % kEntries;
#endif
    }

    //returns true if the thread had not already stopped reporting
    bool stop_reporting() {
      auto id = thread_id();
      auto index = thread_index(id);
      //are we already in this thread?
      if (id == used_threads_[index]) {
        return false;
      }
      used_threads_mutex_[index].lock();
      used_threads_[index] = id;
      return true;
    }

    void start_reporting() {
      auto id = thread_id();
      auto index = thread_index(id);
      auto& v = used_threads_[index];
      if (v == static_cast<entry_type>(index + 1)) {
        return;
      }
      assert(v == id);
      v = index + 1;
      used_threads_mutex_[index].unlock();
    }
  };

  static ThreadTracker& getTracker() {
    static ThreadTracker s_tracker;
    return s_tracker;
  }

}  // namespace

using namespace cms::perftools;

extern "C" {
void alloc_monitor_start() { alloc_monitor_running_state() = true; }
void alloc_monitor_stop() { alloc_monitor_running_state() = false; }

bool alloc_monitor_stop_thread_reporting() { return getTracker().stop_reporting(); }

void alloc_monitor_start_thread_reporting() { getTracker().start_reporting(); }

//----------------------------------------------------------------
//C memory functions

#ifdef USE_LOCAL_MALLOC
void* malloc(size_t size) noexcept {
  const auto original = Originals::malloc;
  Originals::init();
  if (not alloc_monitor_running_state()) {
    return original(size);
  }
  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, original]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}

void* calloc(size_t nitems, size_t item_size) noexcept {
  const auto original = Originals::calloc;
  Originals::init();
  if (not alloc_monitor_running_state()) {
    return original(nitems, item_size);
  }
  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      nitems * item_size,
      [nitems, item_size, original]() { return original(nitems, item_size); },
      [](auto ret) { return malloc_usable_size(ret); });
}
#else
void* malloc(size_t size) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::malloc)>("malloc");
  if (not alloc_monitor_running_state()) {
    return original(size);
  }
  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}

void* calloc(size_t nitems, size_t item_size) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::calloc)>("calloc");
  if (not alloc_monitor_running_state()) {
    return original(nitems, item_size);
  }
  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      nitems * item_size,
      [nitems, item_size]() { return original(nitems, item_size); },
      [](auto ret) { return malloc_usable_size(ret); });
}
#endif

void* realloc(void* ptr, size_t size) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::realloc)>("realloc");
  if (not alloc_monitor_running_state()) {
    return original(ptr, size);
  }
  size_t oldsize = malloc_usable_size(ptr);
  void* ret;
  auto& reg = AllocMonitorRegistry::instance();
  {
    //incase this calls malloc/free
    [[maybe_unused]] auto g = reg.makeGuard();
    ret = original(ptr, size);
  }
  size_t used = malloc_usable_size(ret);
  if (used != oldsize) {
    reg.deallocCalled(
        ptr, [](auto) {}, [oldsize](auto) { return oldsize; });
    reg.allocCalled(
        size, []() { return nullptr; }, [used](auto) { return used; });
  }
  return ret;
}

void* aligned_alloc(size_t alignment, size_t size) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::aligned_alloc)>("aligned_alloc");
  if (not alloc_monitor_running_state()) {
    return original(alignment, size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size,
      [alignment, size]() { return original(alignment, size); },
      [](auto ret) { return malloc_usable_size(ret); });
}

//used by tensorflow
int posix_memalign(void** memptr, size_t alignment, size_t size) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::posix_memalign)>("posix_memalign");
  if (not alloc_monitor_running_state()) {
    return original(memptr, alignment, size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  int ret;
  reg.allocCalled(
      size,
      [&ret, memptr, alignment, size]() {
        ret = original(memptr, alignment, size);
        return *memptr;
      },
      [](auto ret) { return malloc_usable_size(ret); });
  return ret;
}

//used by libc
void* memalign(size_t alignment, size_t size) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::memalign)>("memalign");
  if (not alloc_monitor_running_state()) {
    return original(alignment, size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size,
      [alignment, size]() { return original(alignment, size); },
      [](auto ret) { return malloc_usable_size(ret); });
}

void free(void* ptr) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::free)>("free");
  // ignore memory allocated from our static array at startup
  if (not is_local_alloc(ptr)) {
    if (not alloc_monitor_running_state()) {
      original(ptr);
      return;
    }

    auto& reg = AllocMonitorRegistry::instance();
    reg.deallocCalled(
        ptr, [](auto ptr) { original(ptr); }, [](auto ptr) { return malloc_usable_size(ptr); });
  }
}
}  // extern "C"

//----------------------------------------------------------------
//C++ memory functions

#define CPP_MEM_OVERRIDE

#if defined(CPP_MEM_OVERRIDE)
#include <new>

void* operator new(std::size_t size) {
  CMS_SA_ALLOW static const auto original = get<void* (*)(std::size_t)>("_Znwm");
  if (not alloc_monitor_running_state()) {
    return original(size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_Znwm

void operator delete(void* ptr) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*)>("_ZdlPv");
  if (not alloc_monitor_running_state()) {
    original(ptr);
    return;
  }

  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [](auto ptr) { original(ptr); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdlPv

void* operator new[](std::size_t size) {
  CMS_SA_ALLOW static const auto original = get<void* (*)(std::size_t)>("_Znam");
  if (not alloc_monitor_running_state()) {
    return original(size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_Znam

void operator delete[](void* ptr) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*)>("_ZdaPv");

  if (not alloc_monitor_running_state()) {
    original(ptr);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [](auto ptr) { original(ptr); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdaPv

void* operator new(std::size_t size, std::align_val_t al) {
  CMS_SA_ALLOW static const auto original = get<void* (*)(std::size_t, std::align_val_t)>("_ZnwmSt11align_val_t");
  if (not alloc_monitor_running_state()) {
    return original(size, al);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al]() { return original(size, al); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnwmSt11align_val_t

void* operator new[](std::size_t size, std::align_val_t al) {
  CMS_SA_ALLOW static const auto original = get<void* (*)(std::size_t, std::align_val_t)>("_ZnamSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    return original(size, al);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al]() { return original(size, al); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnamSt11align_val_t

void* operator new(std::size_t size, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void* (*)(std::size_t, const std::nothrow_t&) noexcept>("_ZnwmRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, &tag]() { return original(size, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnwmRKSt9nothrow_t

void* operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void* (*)(std::size_t, const std::nothrow_t&) noexcept>("_ZnamRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, &tag]() { return original(size, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnamRKSt9nothrow_t

void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void* (*)(std::size_t, std::align_val_t, const std::nothrow_t&) noexcept>(
          "_ZnwmSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, al, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al, &tag]() { return original(size, al, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnwmSt11align_val_tRKSt9nothrow_t

void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void* (*)(std::size_t, std::align_val_t, const std::nothrow_t&) noexcept>(
          "_ZnamSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, al, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al, &tag]() { return original(size, al, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnamSt11align_val_tRKSt9nothrow_t

void operator delete(void* ptr, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::align_val_t) noexcept>("_ZdlPvSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [al](auto ptr) { original(ptr, al); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdlPvSt11align_val_t

void operator delete[](void* ptr, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::align_val_t) noexcept>("_ZdaPvSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [al](auto ptr) { original(ptr, al); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdaPvSt11align_val_t

void operator delete(void* ptr, std::size_t sz) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::size_t) noexcept>("_ZdlPvm");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [sz](auto ptr) { original(ptr, sz); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdlPvm

void operator delete[](void* ptr, std::size_t sz) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::size_t) noexcept>("_ZdaPvm");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [sz](auto ptr) { original(ptr, sz); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdaPvm

void operator delete(void* ptr, std::size_t sz, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::size_t, std::align_val_t) noexcept>("_ZdlPvmSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [sz, al](auto ptr) { original(ptr, sz, al); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdlPvmSt11align_val_t

void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::size_t, std::align_val_t) noexcept>("_ZdaPvmSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [sz, al](auto ptr) { original(ptr, sz, al); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdaPvmSt11align_val_t

void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, const std::nothrow_t&) noexcept>("_ZdlPvRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [&tag](auto ptr) { original(ptr, tag); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdlPvRKSt9nothrow_t

void operator delete[](void* ptr, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, const std::nothrow_t&) noexcept>("_ZdaPvRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [&tag](auto ptr) { original(ptr, tag); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdaPvRKSt9nothrow_t

void operator delete(void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::align_val_t, const std::nothrow_t&) noexcept>("_ZdlPvSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [al, &tag](auto ptr) { original(ptr, al, tag); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdlPvSt11align_val_tRKSt9nothrow_t

void operator delete[](void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::align_val_t, const std::nothrow_t&) noexcept>("_ZdaPvSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled(
      ptr, [al, &tag](auto ptr) { original(ptr, al, tag); }, [](auto ptr) { return malloc_usable_size(ptr); });
}  //_ZdaPvSt11align_val_tRKSt9nothrow_t

#endif
