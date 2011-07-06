#ifndef EVF_UTILITIES_TRIGGERREPORTDEF
#define EVF_UTILITIES_TRIGGERREPORTDEF

#include "FWCore/Framework/interface/TriggerReport.h"

namespace evf{
  
  static const size_t max_paths = 500;
  static const size_t max_endpaths = 30;
  static const size_t max_label = 30;
  static const size_t max_modules = 50;

  class ShmOutputModuleRegistry;

  struct ModuleInPathsSummaryStatic{
    //max length of a module label is 80 characters - name is truncated otherwise
    int timesVisited;
    int timesPassed;
    int timesFailed;
    int timesExcept;
    char moduleLabel[max_label];
  };
  struct PathSummaryStatic
  {
    //max length of a path name is 80 characters - name is truncated otherwise
    //max modules in a path are 100
    //    int bitPosition;
    int timesRun;
    int timesPassedPs;
    int timesPassedL1;
    int timesPassed;
    int timesFailed;
    int timesExcept;
    //    int modulesInPath;
    //    char name[max_label];
    //    ModuleInPathsSummaryStatic moduleInPathSummaries[max_modules];
  };
  struct TriggerReportStatic{
    //max number of paths in a menu is 500
    //max number of endpaths in a menu is 20
    unsigned int           lumiSection;
    unsigned int           prescaleIndex;
    edm::EventSummary      eventSummary;
    int                    trigPathsInMenu;
    int                    endPathsInMenu;
    PathSummaryStatic      trigPathSummaries[max_paths];
    PathSummaryStatic      endPathSummaries[max_endpaths];
    TriggerReportStatic();
  };
  namespace funcs{
    void reset(TriggerReportStatic *);
    void addToReport(TriggerReportStatic *trs, TriggerReportStatic *trp, unsigned int lumisection);  
  }
}


#endif
