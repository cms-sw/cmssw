#ifndef FWCore_Framework_SystemTriggerResultsKeeper_h
#define FWCore_Framework_SystemTriggerResultsKeeper_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SystemTriggerReportKeeper
//
/**\class SystemTriggerReportKeeper SystemTriggerReportKeeper.h "SystemTriggerReportKeeper.h"

 Description: Runs timers for system components

 Usage:
    This class is used to keep the results that is used to generate
 the system trigger report.

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

// forward declarations

namespace edm {
  class ModuleDescription;
  class StreamID;
  class StreamContext;
  class PathContext;
  class HLTPathStatus;
  class ModuleCallingContext;
  class ProcessContext;
  struct TriggerReport;
  namespace service {
    class TriggerNamesService;
  }

  class SystemTriggerReportKeeper {
  public:
    SystemTriggerReportKeeper(unsigned int iNumStreams,
                               std::vector<const ModuleDescription*> const& iModules,
                               service::TriggerNamesService const& iNameService,
                               ProcessContext const* iProcessContext);

    SystemTriggerReportKeeper(const SystemTriggerReportKeeper&) = delete;
    SystemTriggerReportKeeper& operator=(const SystemTriggerReportKeeper&) = delete;

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void removeModuleIfExists(ModuleDescription const& module);

    void stopPath(StreamContext const&, PathContext const&, HLTPathStatus const&);

    void stopModuleEvent(StreamContext const&, ModuleCallingContext const&);

    struct ModuleInPathStatus {
      unsigned int m_timesVisited = 0;
      unsigned int m_timesPassed = 0;
      unsigned int m_timesFailed = 0;
      unsigned int m_timesExcept = 0;
    };
    struct PathStatus {
      unsigned int m_timesRun = 0;
      unsigned int m_timesPassed = 0;
      unsigned int m_timesFailed = 0;
      unsigned int m_timesExcept = 0;
      std::vector<ModuleInPathStatus> m_moduleStatus;
    };

    struct ModuleStatus {
      unsigned int m_timesRun = 0;  //TODO if acquire throws, still should mark as run
      unsigned int m_timesPassed = 0;
      unsigned int m_timesFailed = 0;
      unsigned int m_timesExcept = 0;
    };

    void fillTriggerReport(TriggerReport& rep) const;

  private:
    PathStatus& pathStatus(StreamContext const&, PathContext const&);
    bool checkBounds(unsigned int id) const;

    // ---------- member data --------------------------------

    std::vector<std::vector<PathStatus>> m_streamPathStatus;

    std::vector<std::vector<ModuleStatus>> m_streamModuleStatus;

    std::vector<const ModuleDescription*> m_modules;
    std::vector<std::string> m_pathNames;
    std::vector<std::vector<std::string>> m_modulesOnPaths;

    ProcessContext const* m_processContext;

    unsigned int m_minModuleID;
    unsigned int m_endPathOffset;
  };
}  // namespace edm

#endif
