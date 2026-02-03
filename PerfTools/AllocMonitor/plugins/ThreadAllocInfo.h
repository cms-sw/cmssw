#ifndef PerfTools_AllocMonitor_ThreadAllocInfo_h
#define PerfTools_AllocMonitor_ThreadAllocInfo_h
// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     ThreadAllocInfo
//
/**\class ThreadAllocInfo ThreadAllocInfo.h "ThreadAllocInfo.h"

 Description: information about per module allocations

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 10 May 2024 14:48:59 GMT
//

// system include files

// user include files

// forward declarations

namespace edm::service::moduleAlloc {
  struct ThreadAllocInfo {
    size_t requested_ = 0;
    long long presentActual_ = 0;
    size_t maxActual_ = 0;
    long long minActual_ = 0;
    size_t maxSingleAlloc_ = 0;
    size_t nAllocations_ = 0;
    size_t nDeallocations_ = 0;
    bool active_ = false;

    void reset() {
      requested_ = 0;
      presentActual_ = 0;
      maxActual_ = 0;
      minActual_ = 0;
      maxSingleAlloc_ = 0;
      nAllocations_ = 0;
      nDeallocations_ = 0;
      active_ = true;
    }

    void activate() { active_ = true; }
    void deactivate() { active_ = false; }
  };
}  // namespace edm::service::moduleAlloc
#endif
