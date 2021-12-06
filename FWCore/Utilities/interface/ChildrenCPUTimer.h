#ifndef FWCore_Utilities_ChildrenCPUTimer_h
#define FWCore_Utilities_ChildrenCPUTimer_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ChildrenCPUTimer
//
/**\class ChildrenCPUTimer ChildrenCPUTimer.h FWCore/Utilities/interface/ChildrenCPUTimer.h

 Description: Timer which measures the CPU time for child processes

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Apr 16 20:32:13 EDT 2006
//

// system include files
#include <sys/time.h>

// user include files

// forward declarations
namespace edm {
  class ChildrenCPUTimer {
  public:
    ChildrenCPUTimer();
    ~ChildrenCPUTimer();
    ChildrenCPUTimer(ChildrenCPUTimer&&) = default;
    ChildrenCPUTimer(const ChildrenCPUTimer&) = delete;
    ChildrenCPUTimer& operator=(const ChildrenCPUTimer&) = delete;

    // ---------- const member functions ---------------------
    double cpuTime() const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void start();
    double stop();  //returns delta time
    void reset();

    void add(double t);

  private:
    double calculateDeltaTime() const;

    // ---------- member data --------------------------------
    enum State { kRunning, kStopped } state_;
    struct timeval startCPUTime_;

    double accumulatedCPUTime_;
  };
}  // namespace edm

#endif
