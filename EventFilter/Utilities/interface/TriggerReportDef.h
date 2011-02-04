#ifndef EVF_UTILITIES_TRIGGERREPORTDEF
#define EVF_UTILITIES_TRIGGERREPORTDEF

#include "FWCore/Framework/interface/TriggerReport.h"

namespace evf{
  
  static const size_t max_paths = 300;
  static const size_t max_endpaths = 10;
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
    void reset(){
      lumiSection = 0;
      prescaleIndex = 0;
      //copy the event summary
      eventSummary.totalEvents = 0;
      eventSummary.totalEventsPassed = 0;
      eventSummary.totalEventsFailed = 0;
      
      for(size_t i = 0; i < max_paths; i++)
	{
	  // reset individual path summaries
	  trigPathSummaries[i].timesRun = 0;
	  trigPathSummaries[i].timesPassed = 0; 
	  trigPathSummaries[i].timesPassedPs = 0; 
	  trigPathSummaries[i].timesPassedL1 = 0; 
	  trigPathSummaries[i].timesFailed = 0;
	  trigPathSummaries[i].timesExcept = 0;
	}
      for(size_t i = 0; i < max_endpaths; i++)
	{
	  endPathSummaries[i].timesRun    = 0;
	  endPathSummaries[i].timesPassed = 0;
	  endPathSummaries[i].timesPassedPs = 0; 
	  endPathSummaries[i].timesPassedL1 = 0; 
	  endPathSummaries[i].timesFailed = 0;
	  endPathSummaries[i].timesExcept = 0;
	}
/*       trigPathsInMenu = 0; */
/*       endPathsInMenu  = 0; */
    }
    void addToReport(TriggerReportStatic *trp, unsigned int lumisection){
      if(trigPathsInMenu==0) trigPathsInMenu = trp->trigPathsInMenu;
      if(endPathsInMenu==0) endPathsInMenu = trp->endPathsInMenu;
      // set LS and PS
      lumiSection = lumisection;
      prescaleIndex = trp->prescaleIndex;

      //add to the event summary
      eventSummary.totalEvents += trp->eventSummary.totalEvents;
      eventSummary.totalEventsPassed += trp->eventSummary.totalEventsPassed;
      eventSummary.totalEventsFailed += trp->eventSummary.totalEventsFailed;
      //traverse the trigger report and sum relevant parts, check otherwise
      // loop on paths
      for(int i = 0; i < trp->trigPathsInMenu; i++)
	{
	  
	  // fill individual path summaries
	  trigPathSummaries[i].timesRun += trp->trigPathSummaries[i].timesRun;
	  trigPathSummaries[i].timesPassed += trp->trigPathSummaries[i].timesPassed;
	  trigPathSummaries[i].timesPassedPs += trp->trigPathSummaries[i].timesPassedPs;
	  trigPathSummaries[i].timesPassedL1 += trp->trigPathSummaries[i].timesPassedL1;
	  trigPathSummaries[i].timesFailed += trp->trigPathSummaries[i].timesFailed; 
	  trigPathSummaries[i].timesExcept += trp->trigPathSummaries[i].timesExcept;
	}
      for(int i = 0; i < trp->endPathsInMenu; i++)
	{
	  endPathSummaries[i].timesRun += trp->endPathSummaries[i].timesRun;
	  endPathSummaries[i].timesPassed += trp->endPathSummaries[i].timesPassed;
	  endPathSummaries[i].timesPassedPs += trp->endPathSummaries[i].timesPassedPs;
	  endPathSummaries[i].timesPassedL1 += trp->endPathSummaries[i].timesPassedL1;
	  endPathSummaries[i].timesFailed += trp->endPathSummaries[i].timesFailed;
	  endPathSummaries[i].timesExcept += trp->endPathSummaries[i].timesExcept;
	}
    }
  };
}


#endif
