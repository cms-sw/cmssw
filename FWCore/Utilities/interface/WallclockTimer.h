#ifndef FWCore_Utilities_WallclockTimer_h
#define FWCore_Utilities_WallclockTimer_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     WallclockTimer
// 
/**\class WallclockTimer WallclockTimer.h FWCore/Utilities/interface/WallclockTimer.h

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
  class WallclockTimer
  {
    
  public:
    WallclockTimer();
    ~WallclockTimer();
    WallclockTimer(WallclockTimer&&) = default;
    
    // ---------- const member functions ---------------------
    double realTime() const ;
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    void start();
    double stop(); //returns delta time
    void reset();
    
    void add(double t);
  private:
    WallclockTimer(const WallclockTimer&) = delete; // stop default
    
    const WallclockTimer& operator=(const WallclockTimer&) = delete; // stop default
    
    double calculateDeltaTime() const;
    
    // ---------- member data --------------------------------
    enum State {kRunning, kStopped} state_;
#ifdef USE_CLOCK_GETTIME
    struct timespec startRealTime_;
#else
    struct timeval startRealTime_;
#endif
    
    double accumulatedRealTime_;
    
  };
}

#endif
