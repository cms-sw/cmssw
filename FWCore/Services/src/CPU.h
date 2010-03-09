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

#include "sigc++/signal.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

namespace edm {
  struct ActivityRegistry;
  class Event;
  class EventSetup;
  class ParameterSet;
  class ConfigurationDescriptions;

  namespace service {
    class CPU
    {
    public:
      CPU(const ParameterSet&,ActivityRegistry&);
      ~CPU();

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

      sigc::signal<void, const ModuleDescription&, double> newMeasurementSignal;
    private:
      int totalNumberCPUs_;
      double averageCoreSpeed_;
      bool reportCPUProperties_;

      void postEndJob();
    };
  }
}
#endif
