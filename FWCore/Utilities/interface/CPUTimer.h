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
// $Id: CPUTimer.h,v 1.4 2010/10/30 01:30:57 chrjones Exp $
//

// system include files
#ifdef __linux__
//NOTE: clock_gettime is not available on OS X and is slower
// than getrusage and gettimeofday on linux but gives greater
// timing accuracy so we may want to revisit this in the future
//#define USE_CLOCK_GETTIME
#endif

#ifdef USE_CLOCK_GETTIME
#include <time.h>
#else
#include <sys/time.h>
#endif

// user include files

// forward declarations
namespace edm {
class CPUTimer
{

   public:
      CPUTimer();
      virtual ~CPUTimer();

      struct Times {
         Times():real_(0),cpu_(0) {}
         double real_;
         double cpu_;
      };
   
   
      // ---------- const member functions ---------------------
      double realTime() const ;
      double cpuTime() const ;
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void start();
      Times stop(); //returns delta time
      void reset();
      
      void add(const Times& t);
   private:
      CPUTimer(const CPUTimer&); // stop default

      const CPUTimer& operator=(const CPUTimer&); // stop default

      Times calculateDeltaTime() const;
      
      // ---------- member data --------------------------------
      enum State {kRunning, kStopped} state_;
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
}

#endif
