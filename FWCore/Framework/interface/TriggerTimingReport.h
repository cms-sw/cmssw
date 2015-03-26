#ifndef FWCore_Framework_TriggerTimingReport_h
#define FWCore_Framework_TriggerTimingReport_h

/*----------------------------------------------------------------------

TriggerTimingReport: This struct contains all the information relevant to
reporting on the timing of the trigger.


----------------------------------------------------------------------*/

#include <string>
#include <vector>

namespace edm {

  struct EventTimingSummary
  {
    int totalEvents = 0;
    double cpuTime = 0.;
    double realTime =0.;
    double sumStreamRealTime = 0.;
  };

  struct ModuleInPathTimingSummary
  {
    int timesVisited = 0;
    double realTime =0.;

    std::string moduleLabel;
  };


  struct PathTimingSummary
  {
    int bitPosition = 0;
    int timesRun = 0;
    double realTime =0.;

    std::string name;
    std::vector<ModuleInPathTimingSummary> moduleInPathSummaries;
  };

  struct WorkerTimingSummary
  {
    int timesVisited = 0;
    int timesRun = 0;
    double realTime =0.;

    std::string moduleLabel;
  };


  struct TriggerTimingReport
  {
    EventTimingSummary               eventSummary;
    std::vector<PathTimingSummary>   trigPathSummaries;
    std::vector<PathTimingSummary>   endPathSummaries;
    std::vector<WorkerTimingSummary> workerSummaries;
  };

}
#endif
