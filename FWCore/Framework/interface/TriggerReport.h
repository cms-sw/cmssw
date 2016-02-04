#ifndef FWCore_Framework_TriggerReport_h
#define FWCore_Framework_TriggerReport_h

/*----------------------------------------------------------------------

TriggerReport: This struct contains all the information relevant to
reporting on the behavior of the trigger.ed at the time of its
creation.

$Id: TriggerReport.h,v 1.2 2007/06/14 17:52:16 wmtan Exp $

----------------------------------------------------------------------*/

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
