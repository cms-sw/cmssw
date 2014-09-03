#ifndef FWCore_Framework_SystemTimeKeeper_h
#define FWCore_Framework_SystemTimeKeeper_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SystemTimeKeeper
// 
/**\class SystemTimeKeeper SystemTimeKeeper.h "SystemTimeKeeper.h"

 Description: Runs timers for system components

 Usage:
    This class is used to keep the time that is used to generate
 the system time report.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 07 Jul 2014 14:37:31 GMT
//

// system include files
#include <atomic>
#include <vector>
#include <string>

// user include files
#include "FWCore/Utilities/interface/CPUTimer.h"

// forward declarations

namespace edm {
  class ModuleDescription;
  class StreamID;
  class StreamContext;
  class PathContext;
  class HLTPathStatus;
  class ModuleCallingContext;
  class TriggerTimingReport;
  namespace service {
    class TriggersNameService;
  }
  
  class SystemTimeKeeper
  {
    
  public:
    SystemTimeKeeper(unsigned int iNumStreams,
                     std::vector<const ModuleDescription*> const& iModules,
                     service::TriggerNamesService const& iNameService);
    
    // ---------- const member functions ---------------------
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    void startEvent(StreamID);
    void stopEvent(StreamContext const&);
    
    void startPath(StreamContext const&, PathContext const&);
    void stopPath(StreamContext const&, PathContext const&, HLTPathStatus const&);
    
    void startModuleEvent(StreamContext const&, ModuleCallingContext const&);
    void stopModuleEvent(StreamContext const&, ModuleCallingContext const&);
    void pauseModuleEvent(StreamContext const&, ModuleCallingContext const&);
    void restartModuleEvent(StreamContext const&, ModuleCallingContext const&);
    
    struct ModuleInPathTiming {
      double m_realTime = 0.;
      double m_cpuTime = 0.;
      unsigned int m_timesVisited = 0;
    };
    struct PathTiming {
      CPUTimer m_timer;
      std::vector<ModuleInPathTiming> m_moduleTiming;
    };

    struct ModuleTiming {
      CPUTimer m_timer;
      unsigned int m_timesRun =0;
    };

    void fillTriggerTimingReport( TriggerTimingReport& rep) ;
  private:
    SystemTimeKeeper(const SystemTimeKeeper&) = delete; // stop default
    
    const SystemTimeKeeper& operator=(const SystemTimeKeeper&) = delete; // stop default
    
    PathTiming& pathTiming(StreamContext const&, PathContext const&);
    
    // ---------- member data --------------------------------
    std::vector<CPUTimer> m_streamEventTimer;
    
    std::vector<std::vector<PathTiming>> m_streamPathTiming;
    
    std::vector<std::vector<ModuleTiming>> m_streamModuleTiming;
    
    std::vector<const ModuleDescription*>  m_modules;
    std::vector<std::string> m_pathNames;
    std::vector<std::vector<std::string>> m_modulesOnPaths;
    
    unsigned int m_minModuleID;
    unsigned int m_endPathOffset;
    std::atomic<unsigned int> m_numberOfEvents;

  };
}


#endif
