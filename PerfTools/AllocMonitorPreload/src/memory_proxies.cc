#include <memory>
#include <cassert>
#include <atomic>
#include <malloc.h>

#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

#include <dlfcn.h>  // dlsym

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

}  // namespace

using namespace cms::perftools;
extern "C" {
void alloc_monitor_start() { alloc_monitor_running_state() = true; }
void alloc_monitor_stop() { alloc_monitor_running_state() = false; }

//----------------------------------------------------------------
//C memory functions

void* malloc(size_t size) noexcept {
  static auto original = get<decltype(&::malloc)>("malloc");
  if (not alloc_monitor_running_state()) {
    return original(size);
  }
  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}

void* calloc(size_t nitems, size_t item_size) noexcept {
  static auto original = get<decltype(&::calloc)>("calloc");

  if (not alloc_monitor_running_state()) {
    return original(nitems, item_size);
  }
  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      nitems * item_size,
      [nitems, item_size]() { return original(nitems, item_size); },
      [](auto ret) { return malloc_usable_size(ret); });
}

void* realloc(void* ptr, size_t size) noexcept {
  static auto original = get<decltype(&::realloc)>("realloc");
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
  static auto original = get<decltype(&::aligned_alloc)>("aligned_alloc");
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
  static auto original = get<decltype(&::free)>("free");
  if (not alloc_monitor_running_state()) {
    original(ptr);
    return;
  }

  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr]() { original(ptr); }, [ptr]() { return malloc_usable_size(ptr); });
}
}

//----------------------------------------------------------------
//C++ memory functions

#define CPP_MEM_OVERRIDE

#if defined(CPP_MEM_OVERRIDE)
#include <new>

void* operator new(std::size_t size) {
  static auto original = get<void* (*)(std::size_t)>("_Znwm");
  if (not alloc_monitor_running_state()) {
    return original(size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_Znwm

void operator delete(void* ptr) noexcept {
  static auto original = get<void (*)(void*)>("_ZdlPv");
  if (not alloc_monitor_running_state()) {
    original(ptr);
    return;
  }

  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr]() { original(ptr); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPv

void* operator new[](std::size_t size) {
  static auto original = get<void* (*)(std::size_t)>("_Znam");
  if (not alloc_monitor_running_state()) {
    return original(size);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size]() { return original(size); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_Znam

void operator delete[](void* ptr) noexcept {
  static auto original = get<void (*)(void*)>("_ZdaPv");

  if (not alloc_monitor_running_state()) {
    original(ptr);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr]() { original(ptr); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPv

void* operator new(std::size_t size, std::align_val_t al) {
  static auto original = get<void* (*)(std::size_t, std::align_val_t)>("_ZnwmSt11align_val_t");
  if (not alloc_monitor_running_state()) {
    return original(size, al);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al]() { return original(size, al); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnwmSt11align_val_t

void* operator new[](std::size_t size, std::align_val_t al) {
  static auto original = get<void* (*)(std::size_t, std::align_val_t)>("_ZnamSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    return original(size, al);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al]() { return original(size, al); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnamSt11align_val_t

void* operator new(std::size_t size, const std::nothrow_t& tag) noexcept {
  static auto original = get<void* (*)(std::size_t, const std::nothrow_t&) noexcept>("_ZnwmRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, &tag]() { return original(size, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnwmRKSt9nothrow_t

void* operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
  static auto original = get<void* (*)(std::size_t, const std::nothrow_t&) noexcept>("_ZnamRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, &tag]() { return original(size, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnamRKSt9nothrow_t

void* operator new(std::size_t size, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  static auto original = get<void* (*)(std::size_t, std::align_val_t, const std::nothrow_t&) noexcept>(
      "_ZnwmSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, al, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al, &tag]() { return original(size, al, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnwmSt11align_val_tRKSt9nothrow_t

void* operator new[](std::size_t size, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  static auto original = get<void* (*)(std::size_t, std::align_val_t, const std::nothrow_t&) noexcept>(
      "_ZnamSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    return original(size, al, tag);
  }

  auto& reg = AllocMonitorRegistry::instance();
  return reg.allocCalled(
      size, [size, al, &tag]() { return original(size, al, tag); }, [](auto ret) { return malloc_usable_size(ret); });
}  //_ZnamSt11align_val_tRKSt9nothrow_t

void operator delete(void* ptr, std::align_val_t al) noexcept {
  static auto original = get<void (*)(void*, std::align_val_t) noexcept>("_ZdlPvSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al]() { original(ptr, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvSt11align_val_t

void operator delete[](void* ptr, std::align_val_t al) noexcept {
  static auto original = get<void (*)(void*, std::align_val_t) noexcept>("_ZdaPvSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al]() { original(ptr, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvSt11align_val_t

void operator delete(void* ptr, std::size_t sz) noexcept {
  static auto original = get<void (*)(void*, std::size_t) noexcept>("_ZdlPvm");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz]() { original(ptr, sz); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvm

void operator delete[](void* ptr, std::size_t sz) noexcept {
  static auto original = get<void (*)(void*, std::size_t) noexcept>("_ZdaPvm");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz]() { original(ptr, sz); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvm

void operator delete(void* ptr, std::size_t sz, std::align_val_t al) noexcept {
  static auto original = get<void (*)(void*, std::size_t, std::align_val_t) noexcept>("_ZdlPvmSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz, al]() { original(ptr, sz, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvmSt11align_val_t
void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) noexcept {
  static auto original = get<void (*)(void*, std::size_t, std::align_val_t) noexcept>("_ZdaPvmSt11align_val_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, sz, al);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, sz, al]() { original(ptr, sz, al); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvmSt11align_val_t

void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
  static auto original = get<void (*)(void*, const std::nothrow_t&) noexcept>("_ZdlPvRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, &tag]() { original(ptr, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvRKSt9nothrow_t
void operator delete[](void* ptr, const std::nothrow_t& tag) noexcept {
  static auto original = get<void (*)(void*, const std::nothrow_t&) noexcept>("_ZdaPvRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, &tag]() { original(ptr, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvRKSt9nothrow_t

void operator delete(void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  static auto original =
      get<void (*)(void*, std::align_val_t, const std::nothrow_t&) noexcept>("_ZdlPvSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al, &tag]() { original(ptr, al, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdlPvSt11align_val_tRKSt9nothrow_t
void operator delete[](void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept {
  static auto original =
      get<void (*)(void*, std::align_val_t, const std::nothrow_t&) noexcept>("_ZdaPvSt11align_val_tRKSt9nothrow_t");

  if (not alloc_monitor_running_state()) {
    original(ptr, al, tag);
    return;
  }
  auto& reg = AllocMonitorRegistry::instance();
  reg.deallocCalled([ptr, al, &tag]() { original(ptr, al, tag); }, [ptr]() { return malloc_usable_size(ptr); });
}  //_ZdaPvSt11align_val_tRKSt9nothrow_t

#endif
