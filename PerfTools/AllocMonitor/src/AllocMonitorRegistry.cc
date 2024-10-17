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
bool alloc_monitor_stop_thread_reporting();
void alloc_monitor_start_thread_reporting();
}

namespace {
  bool& dummyThreadReportingFcn() {
    static thread_local bool s_running = true;
    return s_running;
  }

  bool dummyStopThreadReportingFcn() {
    auto& t = dummyThreadReportingFcn();
    auto last = t;
    t = false;
    return last;
  }

  void dummyStartThreadReportingFcn() { dummyThreadReportingFcn() = true; }

  bool stopThreadReporting() {
    static decltype(&::alloc_monitor_stop_thread_reporting) s_fcn = []() {
      void* fcn = dlsym(RTLD_DEFAULT, "alloc_monitor_stop_thread_reporting");
      if (fcn != nullptr) {
        return reinterpret_cast<decltype(&::alloc_monitor_stop_thread_reporting)>(fcn);
      }
      //this should only be called for testing;
      return &::dummyStopThreadReportingFcn;
    }();
    return s_fcn();
  }

  void startThreadReporting() {
    static decltype(&::alloc_monitor_start_thread_reporting) s_fcn = []() {
      void* fcn = dlsym(RTLD_DEFAULT, "alloc_monitor_start_thread_reporting");
      if (fcn != nullptr) {
        return reinterpret_cast<decltype(&::alloc_monitor_start_thread_reporting)>(fcn);
      }
      //this should only be called for testing;
      return &::dummyStartThreadReportingFcn;
    }();
    s_fcn();
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
  //Cannot start here because statics can cause memory to be allocated in the atexit registration
  // done behind the scenes. If the allocation happens, AllocMonitorRegistry::instance will be called
  // recursively before the static has finished and we well deadlock
  startReporting();
}

AllocMonitorRegistry::~AllocMonitorRegistry() {
  void* stop = dlsym(RTLD_DEFAULT, "alloc_monitor_stop");
  if (stop != nullptr) {
    auto s = reinterpret_cast<decltype(&::alloc_monitor_stop)>(stop);
    s();
  }
  stopReporting();
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

bool AllocMonitorRegistry::stopReporting() { return stopThreadReporting(); }
void AllocMonitorRegistry::startReporting() { startThreadReporting(); }

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
void AllocMonitorRegistry::allocCalled_(size_t iRequested, size_t iActual, void const* iPtr) {
  for (auto& m : monitors_) {
    m->allocCalled(iRequested, iActual, iPtr);
  }
}
void AllocMonitorRegistry::deallocCalled_(size_t iActual, void const* iPtr) {
  for (auto& m : monitors_) {
    m->deallocCalled(iActual, iPtr);
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
