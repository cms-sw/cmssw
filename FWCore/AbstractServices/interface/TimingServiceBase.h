#ifndef FWCore_AbstractServices_interface_TimingServiceBase_h
#define FWCore_AbstractServices_interface_TimingServiceBase_h
// -*- C++ -*-
//
// Package:     FWCore/AbstractServices
// Class  :     TimingServiceBase
//
/**\class TimingServiceBase TimingServiceBase.h "TimingServiceBase.h"

 Description: Base class for Timing Services

 Usage:
    Provides an interface to allow

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 11 Jun 2014 14:50:33 GMT
//

// system include files
#include <chrono>

// user include files
#include "FWCore/Utilities/interface/StreamID.h"

// forward declarations
namespace edm {
  class TimingServiceBase {
  public:
    TimingServiceBase();
    TimingServiceBase(const TimingServiceBase&) = delete;
    const TimingServiceBase& operator=(const TimingServiceBase&) = delete;
    TimingServiceBase(TimingServiceBase&&) = delete;
    const TimingServiceBase& operator=(TimingServiceBase&&) = delete;
    virtual ~TimingServiceBase();

    // ---------- member functions ---------------------------
    ///Extra CPU time used by a job but not seen by cmsRun
    /// The value should be in seconds.
    /// This function is safe to call from multiple threads
    virtual void addToCPUTime(double iTime) = 0;

    ///CPU time used by this process and all its children.
    /// The value returned should be in seconds.
    virtual double getTotalCPU() const = 0;

    static void jobStarted();

    static std::chrono::steady_clock::time_point jobStartTime() { return s_jobStartTime; }

  private:
    static std::chrono::steady_clock::time_point s_jobStartTime;
  };
}  // namespace edm

#endif
