#ifndef AllocMonitor_interface_AllocMonitorBase_h
#define AllocMonitor_interface_AllocMonitorBase_h
// -*- C++ -*-
//
// Package:     AllocMonitor/interface
// Class  :     AllocMonitorBase
//
/**\class AllocMonitorBase AllocMonitorBase.h "AllocMonitorBase.h"

 Description: Base class for extensions that monitor allocations

 Usage:
    The class is required to be thread safe as all member functions
 will be called concurrently when used in a multi-threaded program.

 If allocations are done within the methods, no callbacks will be 
 generated as the underlying system will temporarily suspend such
 calls on the thread running the method.

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 21 Aug 2023 14:03:34 GMT
//

// system include files
#include <stddef.h>  //size_t

// user include files

// forward declarations

namespace cms::perftools {

  class AllocMonitorBase {
  public:
    AllocMonitorBase();
    virtual ~AllocMonitorBase();

    AllocMonitorBase(const AllocMonitorBase&) = delete;             // stop default
    AllocMonitorBase(AllocMonitorBase&&) = delete;                  // stop default
    AllocMonitorBase& operator=(const AllocMonitorBase&) = delete;  // stop default
    AllocMonitorBase& operator=(AllocMonitorBase&&) = delete;       // stop default

    // ---------- member functions ---------------------------
    virtual void allocCalled(size_t iRequestedSize, size_t iActualSize) = 0;
    virtual void deallocCalled(size_t iActualSize) = 0;
  };
}  // namespace cms::perftools
#endif
