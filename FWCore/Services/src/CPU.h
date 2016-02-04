#ifndef Services_CPU_h
#define Services_CPU_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     CPU
//
//
// Original Author:  Natalia Garcia
// CPU.h: v 1.0 2009/01/08 11:27:50
//

namespace edm {
  class ActivityRegistry;
  class Event;
  class EventSetup;
  class ParameterSet;
  class ConfigurationDescriptions;

  namespace service {
    class CPU {
    public:
      CPU(ParameterSet const&, ActivityRegistry&);
      ~CPU();

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

    private:
      int totalNumberCPUs_;
      double averageCoreSpeed_;
      bool reportCPUProperties_;

      void postEndJob();
    };

    inline
    bool isProcessWideService(CPU const*) {
      return true;
    }
  }
}
#endif
