// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     AllocMonitorRegistry
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 21 Aug 2023 15:42:48 GMT
//

// system include files
#include <dlfcn.h>  // dlsym

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

//
// constants, enums and typedefs
//
extern "C" {
void alloc_monitor_start();
void alloc_monitor_stop();
}

namespace {
  bool& threadRunning() {
    static thread_local bool s_running = true;
    return s_running;
  }
}  // namespace

using namespace cms::perftools;

//
// static data member definitions
//

//
// constructors and destructor
//
AllocMonitorRegistry::AllocMonitorRegistry() {
  threadRunning() = true;
  //Cannot start here because statics can cause memory to be allocated in the atexit registration
  // done behind the scenes. If the allocation happens, AllocMonitorRegistry::instance will be called
  // recursively before the static has finished and we well deadlock
}

AllocMonitorRegistry::~AllocMonitorRegistry() {
  void* stop = dlsym(RTLD_DEFAULT, "alloc_monitor_stop");
  if (stop != nullptr) {
    auto s = reinterpret_cast<decltype(&::alloc_monitor_stop)>(stop);
    s();
  }
  threadRunning() = false;
  monitors_.clear();
}

//
// member functions
//
bool AllocMonitorRegistry::necessaryLibraryWasPreloaded() {
  return dlsym(RTLD_DEFAULT, "alloc_monitor_start") != nullptr;
}

void AllocMonitorRegistry::start() {
  if (monitors_.empty()) {
    void* start = dlsym(RTLD_DEFAULT, "alloc_monitor_start");
    if (start == nullptr) {
      throw cms::Exception("NoAllocMonitorPreload")
          << "The libPerfToolsAllocMonitorPreload.so was not LD_PRELOADed into the job";
    }
    auto s = reinterpret_cast<decltype(&::alloc_monitor_start)>(start);
    s();
  }
}

bool& AllocMonitorRegistry::isRunning() { return threadRunning(); }

void AllocMonitorRegistry::deregisterMonitor(AllocMonitorBase* iMonitor) {
  for (auto it = monitors_.begin(); monitors_.end() != it; ++it) {
    if (it->get() == iMonitor) {
      [[maybe_unused]] Guard g = makeGuard();
      monitors_.erase(it);
      break;
    }
  }
}

//
// const member functions
//
void AllocMonitorRegistry::allocCalled_(size_t iRequested, size_t iActual) {
  for (auto& m : monitors_) {
    m->allocCalled(iRequested, iActual);
  }
}
void AllocMonitorRegistry::deallocCalled_(size_t iActual) {
  for (auto& m : monitors_) {
    m->deallocCalled(iActual);
  }
}

//
// static member functions
//
AllocMonitorRegistry& AllocMonitorRegistry::instance() {
  //The thread unsafe methods are marked as unsafe
  CMS_SA_ALLOW static AllocMonitorRegistry s_registry;
  return s_registry;
}
