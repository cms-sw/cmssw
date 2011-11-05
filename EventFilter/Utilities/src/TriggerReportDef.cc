#include "EventFilter/Utilities/interface/TriggerReportDef.h"

namespace evf{
  TriggerReportStatic::TriggerReportStatic() : trigPathsInMenu(0), endPathsInMenu(0){}
  void funcs::reset(TriggerReportStatic *trs){
    trs->lumiSection = 0;
    trs->prescaleIndex = 0;
    trs->nbExpected=0;
    trs->nbReporting=0;
    //copy the event summary
    trs->eventSummary.totalEvents = 0;
    trs->eventSummary.totalEventsPassed = 0;
    trs->eventSummary.totalEventsFailed = 0;
    
    for(size_t i = 0; i < max_paths; i++)
      {
	// reset individual path summaries
	trs->trigPathSummaries[i].timesRun = 0;
	trs->trigPathSummaries[i].timesPassed = 0; 
	trs->trigPathSummaries[i].timesPassedPs = 0; 
	trs->trigPathSummaries[i].timesPassedL1 = 0; 
	trs->trigPathSummaries[i].timesFailed = 0;
	trs->trigPathSummaries[i].timesExcept = 0;
      }
    for(size_t i = 0; i < max_endpaths; i++)
      {
	trs->endPathSummaries[i].timesRun    = 0;
	trs->endPathSummaries[i].timesPassed = 0;
	trs->endPathSummaries[i].timesPassedPs = 0; 
	trs->endPathSummaries[i].timesPassedL1 = 0; 
	trs->endPathSummaries[i].timesFailed = 0;
	trs->endPathSummaries[i].timesExcept = 0;
      }
    //    trigPathsInMenu = 0;
    //    endPathsInMenu  = 0;
  }
  void funcs::addToReport(TriggerReportStatic *trs, TriggerReportStatic *trp, unsigned int lumisection){
    if(trs->trigPathsInMenu==0) trs->trigPathsInMenu = trp->trigPathsInMenu;
    if(trs->endPathsInMenu==0) trs->endPathsInMenu = trp->endPathsInMenu;
    // set LS and PS
    trs->lumiSection = lumisection;
    if(trp->eventSummary.totalEvents!=0) //do not update PS if no events seen
      trs->prescaleIndex = trp->prescaleIndex;
    // expected and reporting are incremented at each update by the corresponding contrib
    trs->nbExpected += trp->nbExpected;
    trs->nbReporting += trp->nbReporting;
    //add to the event summary
    trs->eventSummary.totalEvents += trp->eventSummary.totalEvents;
    trs->eventSummary.totalEventsPassed += trp->eventSummary.totalEventsPassed;
    trs->eventSummary.totalEventsFailed += trp->eventSummary.totalEventsFailed;
    //traverse the trigger report and sum relevant parts, check otherwise
    // loop on paths
    for(int i = 0; i < trp->trigPathsInMenu; i++)
      {
	
	// fill individual path summaries
	trs->trigPathSummaries[i].timesRun += trp->trigPathSummaries[i].timesRun;
	trs->trigPathSummaries[i].timesPassed += trp->trigPathSummaries[i].timesPassed;
	trs->trigPathSummaries[i].timesPassedPs += trp->trigPathSummaries[i].timesPassedPs;
	trs->trigPathSummaries[i].timesPassedL1 += trp->trigPathSummaries[i].timesPassedL1;
	trs->trigPathSummaries[i].timesFailed += trp->trigPathSummaries[i].timesFailed; 
	trs->trigPathSummaries[i].timesExcept += trp->trigPathSummaries[i].timesExcept;
      }
    for(int i = 0; i < trp->endPathsInMenu; i++)
      {
	trs->endPathSummaries[i].timesRun += trp->endPathSummaries[i].timesRun;
	trs->endPathSummaries[i].timesPassed += trp->endPathSummaries[i].timesPassed;
	trs->endPathSummaries[i].timesPassedPs += trp->endPathSummaries[i].timesPassedPs;
	trs->endPathSummaries[i].timesPassedL1 += trp->endPathSummaries[i].timesPassedL1;
	trs->endPathSummaries[i].timesFailed += trp->endPathSummaries[i].timesFailed;
	trs->endPathSummaries[i].timesExcept += trp->endPathSummaries[i].timesExcept;
      }
  }
}
