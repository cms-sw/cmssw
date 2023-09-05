#ifndef PerfTools_AllocMonitor_AllocMonitorRegistry_h
#define PerfTools_AllocMonitor_AllocMonitorRegistry_h
// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     AllocMonitorRegistry
//
/**\class AllocMonitorRegistry AllocMonitorRegistry.h "AllocMonitorRegistry.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 21 Aug 2023 14:12:54 GMT
//

// system include files
#include <memory>
#include <vector>
#include <malloc.h>
#include <stdlib.h>

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"

// forward declarations

namespace cms::perftools {
  class AllocTester;

  class AllocMonitorRegistry {
  public:
    ~AllocMonitorRegistry();

    AllocMonitorRegistry(AllocMonitorRegistry&&) = delete;                  // stop default
    AllocMonitorRegistry(const AllocMonitorRegistry&) = delete;             // stop default
    AllocMonitorRegistry& operator=(const AllocMonitorRegistry&) = delete;  // stop default
    AllocMonitorRegistry& operator=(AllocMonitorRegistry&&) = delete;       // stop default

    // ---------- static member functions --------------------
    static AllocMonitorRegistry& instance();
    static bool necessaryLibraryWasPreloaded();

    // ---------- member functions ---------------------------

    //The functions are not thread safe
    template <typename T, typename... ARGS>
    T* createAndRegisterMonitor(ARGS&&... iArgs);
    void deregisterMonitor(AllocMonitorBase*);

  private:
    friend void* ::malloc(size_t) noexcept;
    friend void* ::calloc(size_t, size_t) noexcept;
    friend void* ::realloc(void*, size_t) noexcept;
    friend void* ::aligned_alloc(size_t, size_t) noexcept;
    friend void ::free(void*) noexcept;

    friend void* ::operator new(std::size_t size);
    friend void* ::operator new[](std::size_t size);
    friend void* ::operator new(std::size_t count, std::align_val_t al);
    friend void* ::operator new[](std::size_t count, std::align_val_t al);
    friend void* ::operator new(std::size_t count, const std::nothrow_t& tag) noexcept;
    friend void* ::operator new[](std::size_t count, const std::nothrow_t& tag) noexcept;
    friend void* ::operator new(std::size_t count, std::align_val_t al, const std::nothrow_t&) noexcept;
    friend void* ::operator new[](std::size_t count, std::align_val_t al, const std::nothrow_t&) noexcept;

    friend void ::operator delete(void* ptr) noexcept;
    friend void ::operator delete[](void* ptr) noexcept;
    friend void ::operator delete(void* ptr, std::align_val_t al) noexcept;
    friend void ::operator delete[](void* ptr, std::align_val_t al) noexcept;
    friend void ::operator delete(void* ptr, std::size_t sz) noexcept;
    friend void ::operator delete[](void* ptr, std::size_t sz) noexcept;
    friend void ::operator delete(void* ptr, std::size_t sz, std::align_val_t al) noexcept;
    friend void ::operator delete[](void* ptr, std::size_t sz, std::align_val_t al) noexcept;
    friend void ::operator delete(void* ptr, const std::nothrow_t& tag) noexcept;
    friend void ::operator delete[](void* ptr, const std::nothrow_t& tag) noexcept;
    friend void ::operator delete(void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept;
    friend void ::operator delete[](void* ptr, std::align_val_t al, const std::nothrow_t& tag) noexcept;

    friend class AllocTester;

    // ---------- member data --------------------------------
    void start();
    bool& isRunning();

    struct Guard {
      explicit Guard(bool& iOriginal) noexcept : address_(&iOriginal), original_(iOriginal) { *address_ = false; }
      ~Guard() { *address_ = original_; }

      bool running() const noexcept { return original_; }

      Guard(Guard const&) = delete;
      Guard(Guard&&) = delete;
      Guard& operator=(Guard const&) = delete;
      Guard& operator=(Guard&&) = delete;

    private:
      bool* address_;
      bool original_;
    };

    Guard makeGuard() { return Guard(isRunning()); }

    void allocCalled_(size_t, size_t);
    void deallocCalled_(size_t);

    template <typename ALLOC, typename ACT>
    auto allocCalled(size_t iRequested, ALLOC iAlloc, ACT iGetActual) {
      [[maybe_unused]] Guard g = makeGuard();
      auto a = iAlloc();
      if (g.running()) {
        allocCalled_(iRequested, iGetActual(a));
      }
      return a;
    }
    template <typename DEALLOC, typename ACT>
    void deallocCalled(DEALLOC iDealloc, ACT iGetActual) {
      [[maybe_unused]] Guard g = makeGuard();
      if (g.running()) {
        deallocCalled_(iGetActual());
      }
      iDealloc();
    }

    AllocMonitorRegistry();
    std::vector<std::unique_ptr<AllocMonitorBase>> monitors_;
  };

  template <typename T, typename... ARGS>
  T* AllocMonitorRegistry::createAndRegisterMonitor(ARGS&&... iArgs) {
    [[maybe_unused]] Guard guard = makeGuard();
    start();

    auto m = std::make_unique<T>(std::forward<ARGS>(iArgs)...);
    auto p = m.get();
    monitors_.push_back(std::move(m));
    return p;
  }
}  // namespace cms::perftools
#endif
