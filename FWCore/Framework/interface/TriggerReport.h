#ifndef Framework_TriggerReport_h
#define Framework_TriggerReport_h

/*----------------------------------------------------------------------

TriggerReport: This struct contains all the information relevant to
reporting on the behavior of the trigger.ed at the time of its
creation.

$Id: TriggerReport.h,v 1.20 2006/05/02 15:50:51 paterno Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <string>
#include <vector>

namespace edm {

  struct EventSummary
  {
    int totalEvents;
    int totalEventsPassed;
    int totalEventsFailed;
  };

  struct ModuleInPathSummary
  {
    int timesVisited;
    int timesPassed;
    int timesFailed;
    int timesExcept;

    std::string moduleLabel;
  };


  struct PathSummary
  {
    int bitPosition;
    int timesRun;
    int timesPassed;
    int timesFailed;
    int timesExcept;

    std::string name;
    std::vector<ModuleInPathSummary> moduleInPathSummaries;
  };

  struct WorkerSummary
  {
    int timesVisited;
    int timesRun;
    int timesPassed;
    int timesFailed;
    int timesExcept;

    std::string moduleLabel;
  };


  struct TriggerReport
  {
    EventSummary               eventSummary;
    std::vector<PathSummary>   trigPathSummaries;
    std::vector<PathSummary>   endPathSummaries;
    std::vector<WorkerSummary> workerSummaries;
  };

}
#endif
