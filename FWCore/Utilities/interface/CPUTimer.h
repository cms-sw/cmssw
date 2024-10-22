#ifndef FWCore_Utilities_CPUTimer_h
#define FWCore_Utilities_CPUTimer_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     CPUTimer
//
/**\class CPUTimer CPUTimer.h FWCore/Utilities/interface/CPUTimer.h

 Description: Timer which measures the CPU and wallclock time

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Apr 16 20:32:13 EDT 2006
//

// system include files
#ifdef __linux__
//clock_gettime is not available on OS X
#define USE_CLOCK_GETTIME
#endif

#ifdef USE_CLOCK_GETTIME
#include <ctime>
#else
#include <sys/time.h>
#endif

// user include files

// forward declarations
namespace edm {
  class CPUTimer {
  public:
    CPUTimer();
    ~CPUTimer();
    CPUTimer(CPUTimer&&) = default;
    CPUTimer(const CPUTimer&) = delete;
    CPUTimer& operator=(const CPUTimer&) = delete;

    struct Times {
      Times() : real_(0), cpu_(0) {}
      double real_;
      double cpu_;
    };

    // ---------- const member functions ---------------------
    double realTime() const;
    double cpuTime() const;

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void start();
    Times stop();  //returns delta time
    void reset();

    void add(const Times& t);

  private:
    Times calculateDeltaTime() const;

    // ---------- member data --------------------------------
    enum State { kRunning, kStopped } state_;
#ifdef USE_CLOCK_GETTIME
    struct timespec startRealTime_;
    struct timespec startCPUTime_;
#else
    struct timeval startRealTime_;
    struct timeval startCPUTime_;
#endif

    double accumulatedRealTime_;
    double accumulatedCPUTime_;
  };
}  // namespace edm

#endif
