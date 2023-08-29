#include <memory>
#include <cassert>
#include <atomic>
#include <cstddef>
#include <malloc.h>

#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <dlfcn.h>  // dlsym

#if !defined(__x86_64__) && !defined(__i386__)
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
}  // namespace

using namespace cms::perftools;

extern "C" {
void alloc_monitor_start() { alloc_monitor_running_state() = true; }
void alloc_monitor_stop() { alloc_monitor_running_state() = false; }

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
    reg.deallocCalled([]() {}, [oldsize]() { return oldsize; });
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

void free(void* ptr) noexcept {
  CMS_SA_ALLOW static const auto original = get<decltype(&::free)>("free");
  // ignore memory allocated from our static array at startup
  if (not is_local_alloc(ptr)) {
    if (not alloc_monitor_running_state()) {
      original(ptr);
      return;
    }

    auto& reg = AllocMonitorRegistry::instance();
    reg.deallocCalled([ptr]() { original(ptr); }, [ptr]() { return malloc_usable_size(ptr); });
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
  reg.deallocCalled([ptr]() { original(ptr); }, [ptr]() { return malloc_usable_size(ptr); });
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
  reg.deallocCalled([ptr]() { original(ptr); }, [ptr]() { return malloc_usable_size(ptr); });
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
  reg.deallocCalled([ptr, al]() { original(ptr, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvSt11align_val_t

void operator delete[](void* ptr, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::align_val_t) noexcept>("_ZdaPvSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al]() { original(ptr, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvSt11align_val_t

void operator delete(void* ptr, std::size_t sz) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::size_t) noexcept>("_ZdlPvm");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz]() { original(ptr, sz); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvm

void operator delete[](void* ptr, std::size_t sz) noexcept {
  CMS_SA_ALLOW static const auto original = get<void (*)(void*, std::size_t) noexcept>("_ZdaPvm");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz]() { original(ptr, sz); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvm

void operator delete(void* ptr, std::size_t sz, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::size_t, std::align_val_t) noexcept>("_ZdlPvmSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz, al]() { original(ptr, sz, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvmSt11align_val_t

void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::size_t, std::align_val_t) noexcept>("_ZdaPvmSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz, al]() { original(ptr, sz, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvmSt11align_val_t

void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, const std::nothrow_t&) noexcept>("_ZdlPvRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, &tag]() { original(ptr, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvRKSt9nothrow_t

void operator delete[](void* ptr, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, const std::nothrow_t&) noexcept>("_ZdaPvRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, &tag]() { original(ptr, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvRKSt9nothrow_t

void operator delete(void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::align_val_t, const std::nothrow_t&) noexcept>("_ZdlPvSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al, &tag]() { original(ptr, al, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvSt11align_val_tRKSt9nothrow_t

void operator delete[](void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  CMS_SA_ALLOW static const auto original =
      get<void (*)(void*, std::align_val_t, const std::nothrow_t&) noexcept>("_ZdaPvSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al, &tag]() { original(ptr, al, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvSt11align_val_tRKSt9nothrow_t

#endif
