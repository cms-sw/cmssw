#ifndef FWCore_Framework_TriggerReport_h
#define FWCore_Framework_TriggerReport_h

/*----------------------------------------------------------------------

TriggerReport: This struct contains all the information relevant to
reporting on the behavior of the trigger.ed at the time of its
creation.


----------------------------------------------------------------------*/

#include <string>
#include <vector>

namespace edm {

  struct EventSummary
  {
    int totalEvents = 0;
    int totalEventsPassed = 0;
    int totalEventsFailed = 0;
  };

  struct ModuleInPathSummary
  {
    int timesVisited = 0;
    int timesPassed = 0;
    int timesFailed = 0;
    int timesExcept = 0;

    std::string moduleLabel;
  };


  struct PathSummary
  {
    int bitPosition = 0;
    int timesRun = 0;
    int timesPassed = 0;
    int timesFailed = 0;
    int timesExcept = 0;

    std::string name;
    std::vector<ModuleInPathSummary> moduleInPathSummaries;
  };

  struct WorkerSummary
  {
    int timesVisited = 0;
    int timesRun = 0;
    int timesPassed = 0;
    int timesFailed = 0;
    int timesExcept = 0;

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
