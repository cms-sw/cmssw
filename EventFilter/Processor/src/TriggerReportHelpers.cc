#include "EventFilter/Utilities/interface/Exception.h"
#include "TriggerReportHelpers.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "xdata/TableIterator.h"

#include <iostream>
#include <sstream>
#include <algorithm>

namespace evf{
  namespace fuep{
const std::string TriggerReportHelpers::columns[5] = {"l1Pass","psPass","pAccept","pExcept","pReject"};
void TriggerReportHelpers::triggerReportToTable(edm::TriggerReport &tr, unsigned int ls, bool lumiComplete)
{

  lumiSectionIndex_ = ls;  
  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    if(l1pos_[i]>=0) {
      l1pre_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed 
	+ (lumiComplete ? - pl1pre_[i]  : l1pre_[i].value_);
      pl1pre_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed;
    }
    else {
      l1pre_[i] = tr.trigPathSummaries[i].timesRun 
	+ (lumiComplete ? - pl1pre_[i] : l1pre_[i].value_);
      pl1pre_[i] = tr.trigPathSummaries[i].timesRun;
    }
    triggerReportAsTable_.setValueAt(i,columns[0],l1pre_[i]);
    if(pspos_[i]>=0) {
      ps_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[pspos_[i]].timesPassed 
	+ (lumiComplete ? - pps_[i] : ps_[i].value_);
      pps_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[pspos_[i]].timesPassed;
    }
    else if(l1pos_[i]>=0) {
      ps_[i] =  tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed
	+ (lumiComplete ? - pps_[i] : ps_[i].value_);
      pps_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed; 
    }
    else {
      ps_[i] = tr.trigPathSummaries[i].timesRun 
	+ (lumiComplete ? - pps_[i] : ps_[i].value_);
      pps_[i] = tr.trigPathSummaries[i].timesRun;
    }
    triggerReportAsTable_.setValueAt(i,columns[1],ps_[i]);
    accept_[i] = tr.trigPathSummaries[i].timesPassed 
	+ (lumiComplete ? - paccept_[i] : accept_[i].value_);
    paccept_[i] = tr.trigPathSummaries[i].timesPassed; 
    triggerReportAsTable_.setValueAt(i,columns[2], accept_[i]);
    except_[i] = tr.trigPathSummaries[i].timesExcept 
	+ (lumiComplete ? - pexcept_[i] : except_[i].value_);
    pexcept_[i] = tr.trigPathSummaries[i].timesExcept;
    triggerReportAsTable_.setValueAt(i,columns[3], except_[i]);
    failed_[i] = tr.trigPathSummaries[i].timesFailed 
	+ (lumiComplete ? - pfailed_[i] : failed_[i].value_);
    pfailed_[i] = tr.trigPathSummaries[i].timesFailed;
    triggerReportAsTable_.setValueAt(i,columns[4], failed_[i]);
  }
}

void TriggerReportHelpers::packedTriggerReportToTable()
{


  TriggerReportStatic *trs = getPackedTriggerReportAsStruct();
  for(int i=0; i<trs->trigPathsInMenu; i++) {
    l1pre_[i] = trs->trigPathSummaries[i].timesPassedL1;
    triggerReportAsTable_.setValueAt(i,columns[0],l1pre_[i]);
    ps_[i] = trs->trigPathSummaries[i].timesPassedPs;
    triggerReportAsTable_.setValueAt(i,columns[1],ps_[i]);
    accept_[i] = trs->trigPathSummaries[i].timesPassed;
    triggerReportAsTable_.setValueAt(i,columns[2], accept_[i]);
    except_[i] = trs->trigPathSummaries[i].timesExcept;
    triggerReportAsTable_.setValueAt(i,columns[3], except_[i]);
    failed_[i] = trs->trigPathSummaries[i].timesFailed;
    triggerReportAsTable_.setValueAt(i,columns[4], failed_[i]);

  }
}


void TriggerReportHelpers::formatReportTable(edm::TriggerReport &tr, 
					     std::vector<edm::ModuleDescription const*>& descs,
					     bool noNukeLegenda)  
{

  if(tableFormatted_) return;
  std::ostringstream ost;
  trp_ = tr;
  resetTriggerReport();
  TriggerReportStatic *trp = (TriggerReportStatic *)cache_->mtext;

  tableFormatted_ = true;
  paths_.resize(tr.trigPathSummaries.size());

  //adjust number of trigger- and end-paths in static structure
  trp->trigPathsInMenu = tr.trigPathSummaries.size();
  trp->endPathsInMenu = tr.endPathSummaries.size();
  ///

  l1pos_.resize(tr.trigPathSummaries.size(),-1);
  pspos_.resize(tr.trigPathSummaries.size(),-1);
  l1pre_.resize(tr.trigPathSummaries.size(),0);
  ps_.resize(tr.trigPathSummaries.size(),0);
  accept_.resize(tr.trigPathSummaries.size(),0);
  except_.resize(tr.trigPathSummaries.size(),0);
  failed_.resize(tr.trigPathSummaries.size(),0);
  pl1pre_.resize(tr.trigPathSummaries.size(),0);
  pps_.resize(tr.trigPathSummaries.size(),0);
  paccept_.resize(tr.trigPathSummaries.size(),0);
  pexcept_.resize(tr.trigPathSummaries.size(),0);
  pfailed_.resize(tr.trigPathSummaries.size(),0);
  triggerReportAsTable_.clear();
  triggerReportAsTable_.addColumn(columns[0],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[1],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[2],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[3],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[4],"unsigned int 32");

  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    xdata::Table::iterator it = triggerReportAsTable_.append();
    paths_[i] = tr.trigPathSummaries[i].name;

    ost << i << "=" << paths_[i] << ", ";
    for(unsigned int j=0;j<tr.trigPathSummaries[i].moduleInPathSummaries.size();j++) {
      std::string label = tr.trigPathSummaries[i].moduleInPathSummaries[j].moduleLabel;
      for(unsigned int k = 0; k < descs.size(); k++)
	{
	  if(descs[k]->moduleLabel() == label) 
	    {
	      if(descs[k]->moduleName() == "HLTLevel1GTSeed") l1pos_[i] = j;
	      if(descs[k]->moduleName() == "HLTPrescaler") pspos_[i] = j;
	    }
	}

    }

  }
  if(noNukeLegenda) pathLegenda_.value_ = ost.str();

}

void TriggerReportHelpers::printReportTable()
{
  std::cout << "report table for LS #" << lumiSectionIndex_ << std::endl;
  triggerReportAsTable_.writeTo(std::cout);
}
void TriggerReportHelpers::printTriggerReport(edm::TriggerReport &tr)
{
  std::ostringstream oss;
  
  oss<<"================================="<<"\n";
  oss<<"== BEGINNING OF TRIGGER REPORT =="<<"\n";
  oss<<"================================="<<"\n";
  oss<<"tr.eventSummary.totalEvents= "<<tr.eventSummary.totalEvents<<"\n" 
     <<"tr.eventSummary.totalEventsPassed= "<<tr.eventSummary.totalEventsPassed<<"\n"
     <<"tr.eventSummary.totalEventsFailed= "<<tr.eventSummary.totalEventsFailed<<"\n";
  
  oss<<"TriggerReport::trigPathSummaries"<<"\n";
  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    oss<<"tr.trigPathSummaries["<<i<<"].bitPosition = "
       <<tr.trigPathSummaries[i].bitPosition <<"\n" 
       <<"tr.trigPathSummaries["<<i<<"].timesRun = "
       <<tr.trigPathSummaries[i].timesRun <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].timesPassed = "
       <<tr.trigPathSummaries[i].timesPassed <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].timesFailed = "
       <<tr.trigPathSummaries[i].timesFailed <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].timesExcept = "
       <<tr.trigPathSummaries[i].timesExcept <<"\n"
       <<"tr.trigPathSummaries["<<i<<"].name = "
       <<tr.trigPathSummaries[i].name <<"\n";
    
    //TriggerReport::trigPathSummaries::moduleInPathSummaries
    for(unsigned int j=0;j<tr.trigPathSummaries[i].moduleInPathSummaries.size();j++) {
      oss<<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesVisited = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesVisited<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesPassed = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesPassed<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesFailed = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesFailed<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesExcept = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].timesExcept<<"\n"
	 <<"tr.trigPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].moduleLabel = "
	 <<tr.trigPathSummaries[i].moduleInPathSummaries[j].moduleLabel<<"\n";
    }
  }
  
  //TriggerReport::endPathSummaries
  for(unsigned int i=0;i<tr.endPathSummaries.size();i++) {
    oss<<"tr.endPathSummaries["<<i<<"].bitPosition = "
       <<tr.endPathSummaries[i].bitPosition <<"\n" 
       <<"tr.endPathSummaries["<<i<<"].timesRun = "
       <<tr.endPathSummaries[i].timesRun <<"\n"
       <<"tr.endPathSummaries["<<i<<"].timesPassed = "
       <<tr.endPathSummaries[i].timesPassed <<"\n"
       <<"tr.endPathSummaries["<<i<<"].timesFailed = "
       <<tr.endPathSummaries[i].timesFailed <<"\n"
       <<"tr.endPathSummaries["<<i<<"].timesExcept = "
       <<tr.endPathSummaries[i].timesExcept <<"\n"
       <<"tr.endPathSummaries["<<i<<"].name = "
       <<tr.endPathSummaries[i].name <<"\n";
    
    //TriggerReport::endPathSummaries::moduleInPathSummaries
    for(unsigned int j=0;j<tr.endPathSummaries[i].moduleInPathSummaries.size();j++) {
      oss<<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesVisited = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesVisited <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesPassed = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesPassed <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesFailed = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesFailed <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].timesExcept = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].timesExcept <<"\n"
	 <<"tr.endPathSummaries["<<i
	 <<"].moduleInPathSummaries["<<j<<"].moduleLabel = "
	 <<tr.endPathSummaries[i].moduleInPathSummaries[j].moduleLabel <<"\n";
    }
  }
  
  //TriggerReport::workerSummaries
  for(unsigned int i=0; i<tr.workerSummaries.size(); i++) {
    oss<<"tr.workerSummaries["<<i<<"].timesVisited = "
       <<tr.workerSummaries[i].timesVisited<<"\n" 
       <<"tr.workerSummaries["<<i<<"].timesRun = "
       <<tr.workerSummaries[i].timesRun<<"\n"
       <<"tr.workerSummaries["<<i<<"].timesPassed = "
       <<tr.workerSummaries[i].timesPassed <<"\n"
       <<"tr.workerSummaries["<<i<<"].timesFailed = "
       <<tr.workerSummaries[i].timesFailed <<"\n"
       <<"tr.workerSummaries["<<i<<"].timesExcept = "
       <<tr.workerSummaries[i].timesExcept <<"\n"
       <<"tr.workerSummaries["<<i<<"].moduleLabel = "
       <<tr.workerSummaries[i].moduleLabel <<"\n";
  }
  
  oss<<"==========================="<<"\n";
  oss<<"== END OF TRIGGER REPORT =="<<"\n";
  oss<<"==========================="<<"\n";
  std::cout << oss.str() << std::endl; 
  //  LOG4CPLUS_DEBUG(getApplicationLogger(),oss.str());
}

void TriggerReportHelpers::packTriggerReport(edm::TriggerReport &tr)
{
  TriggerReportStatic *trp = (TriggerReportStatic *)cache_->mtext;
  trp->lumiSection = lumiSectionIndex_;
  //copy the event summary
  trp->eventSummary.totalEvents = 
    tr.eventSummary.totalEvents - trp_.eventSummary.totalEvents;
  trp->eventSummary.totalEventsPassed = 
    tr.eventSummary.totalEventsPassed - trp_.eventSummary.totalEventsPassed;
  trp->eventSummary.totalEventsFailed = 
    tr.eventSummary.totalEventsFailed - trp_.eventSummary.totalEventsFailed;

  //get total paths in the menu
  trp->trigPathsInMenu = std::min(tr.trigPathSummaries.size(),max_paths);
  trp->endPathsInMenu = std::min(tr.endPathSummaries.size(),max_endpaths);
  //traverse the trigger report to get a copy of relevant parts in the static structure
  // loop on paths

  for(unsigned int i = 0; i < trp->trigPathsInMenu; i++)
    {

      trp->trigPathSummaries[i].timesRun = 
	tr.trigPathSummaries[i].timesRun - trp_.trigPathSummaries[i].timesRun;
      trp->trigPathSummaries[i].timesPassed = 
	tr.trigPathSummaries[i].timesPassed - trp_.trigPathSummaries[i].timesPassed;
      if(l1pos_[i]>=0) {
	trp->trigPathSummaries[i].timesPassedL1 =
	  tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed -
	  trp_.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed;
      }
      else {
	trp->trigPathSummaries[i].timesPassedL1 =
	  tr.trigPathSummaries[i].timesRun - trp_.trigPathSummaries[i].timesRun;
      }
      if(pspos_[i]>=0) {
	trp->trigPathSummaries[i].timesPassedPs =
	  tr.trigPathSummaries[i].moduleInPathSummaries[pspos_[i]].timesPassed -
	  trp_.trigPathSummaries[i].moduleInPathSummaries[pspos_[i]].timesPassed;
      }
      else if(l1pos_[i]>=0) {
	trp->trigPathSummaries[i].timesPassedPs =
	  tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed -
	  trp_.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed;
      }
      else {
	trp->trigPathSummaries[i].timesPassedPs =
	  tr.trigPathSummaries[i].timesRun - trp_.trigPathSummaries[i].timesRun;
      }
      trp->trigPathSummaries[i].timesFailed = 
	tr.trigPathSummaries[i].timesFailed-trp_.trigPathSummaries[i].timesFailed;
      trp->trigPathSummaries[i].timesExcept = 
	tr.trigPathSummaries[i].timesExcept - trp_.trigPathSummaries[i].timesExcept;
    }

  for(unsigned int i = 0; i < trp->endPathsInMenu; i++)
    {

      trp->endPathSummaries[i].timesRun    = 
	tr.endPathSummaries[i].timesRun - trp_.endPathSummaries[i].timesRun;
      trp->endPathSummaries[i].timesPassed = 
	tr.endPathSummaries[i].timesPassed - trp_.endPathSummaries[i].timesPassed;
      trp->endPathSummaries[i].timesFailed = 
	tr.endPathSummaries[i].timesFailed - trp_.endPathSummaries[i].timesFailed;
      trp->endPathSummaries[i].timesExcept = 
	tr.endPathSummaries[i].timesExcept - trp_.endPathSummaries[i].timesExcept;
    }
  trp_ = tr; 
}


void TriggerReportHelpers::sumAndPackTriggerReport(MsgBuf &buf)
{

  TriggerReportStatic *trs = (TriggerReportStatic *)cache_->mtext;
  TriggerReportStatic *trp = (TriggerReportStatic *)buf->mtext;

  // add check for LS consistency
  trs->lumiSection = trp->lumiSection;
  //add to the event summary
  trs->eventSummary.totalEvents += trp->eventSummary.totalEvents;
  trs->eventSummary.totalEventsPassed += trp->eventSummary.totalEventsPassed;
  trs->eventSummary.totalEventsFailed += trp->eventSummary.totalEventsFailed;

  //get total paths in the menu
  if(trs->trigPathsInMenu != trp->trigPathsInMenu) 
    XCEPT_RAISE(evf::Exception,"trig summary inconsistency");
  if(trs->endPathsInMenu != trp->endPathsInMenu)
    XCEPT_RAISE(evf::Exception,"trig summary inconsistency");

  //traverse the trigger report and sum relevant parts, check otherwise
  // loop on paths
  for(unsigned int i = 0; i < trp->trigPathsInMenu; i++)
    {

      // fill individual path summaries
      trs->trigPathSummaries[i].timesRun += trp->trigPathSummaries[i].timesRun;
      trs->trigPathSummaries[i].timesPassed += trp->trigPathSummaries[i].timesPassed;
      trs->trigPathSummaries[i].timesPassedPs += trp->trigPathSummaries[i].timesPassedPs;
      trs->trigPathSummaries[i].timesPassedL1 += trp->trigPathSummaries[i].timesPassedL1;
      trs->trigPathSummaries[i].timesFailed += trp->trigPathSummaries[i].timesFailed; 
      trs->trigPathSummaries[i].timesExcept += trp->trigPathSummaries[i].timesExcept;
    }
  for(unsigned int i = 0; i < trp->endPathsInMenu; i++)
    {

      trs->endPathSummaries[i].timesRun += trp->endPathSummaries[i].timesRun;
      trs->endPathSummaries[i].timesPassed += trp->endPathSummaries[i].timesPassed;
      trs->endPathSummaries[i].timesFailed += trp->endPathSummaries[i].timesFailed;
      trs->endPathSummaries[i].timesExcept += trp->endPathSummaries[i].timesExcept;
    }
  
}  

void TriggerReportHelpers::resetPackedTriggerReport()
{

  TriggerReportStatic *trp = (TriggerReportStatic *)cache_->mtext;
  trp->lumiSection = 0;
  //copy the event summary
  trp->eventSummary.totalEvents = 0;
  trp->eventSummary.totalEventsPassed = 0;
  trp->eventSummary.totalEventsFailed = 0;

  for(unsigned int i = 0; i < trp->trigPathsInMenu; i++)
    {
      // reset individual path summaries
      trp->trigPathSummaries[i].timesRun = 0;
      trp->trigPathSummaries[i].timesPassed = 0; 
      trp->trigPathSummaries[i].timesPassedPs = 0; 
      trp->trigPathSummaries[i].timesPassedL1 = 0; 
      trp->trigPathSummaries[i].timesFailed = 0;
      trp->trigPathSummaries[i].timesExcept = 0;
    }
  for(unsigned int i = 0; i < trp->endPathsInMenu; i++)
    {
      trp->endPathSummaries[i].timesRun    = 0;
      trp->endPathSummaries[i].timesPassed = 0;
      trp->endPathSummaries[i].timesFailed = 0;
      trp->endPathSummaries[i].timesExcept = 0;
    }

}

void TriggerReportHelpers::resetTriggerReport()
{

  //copy the event summary
  trp_.eventSummary.totalEvents = 0;
  trp_.eventSummary.totalEventsPassed = 0;
  trp_.eventSummary.totalEventsFailed = 0;

  for(unsigned int i = 0; i < trp_.trigPathSummaries.size(); i++)
    {
      // reset individual path summaries
      trp_.trigPathSummaries[i].timesRun = 0;
      trp_.trigPathSummaries[i].timesPassed = 0; 
      trp_.trigPathSummaries[i].timesFailed = 0;
      trp_.trigPathSummaries[i].timesExcept = 0;

      //loop over modules in path
      for(unsigned int j = 0; j<trp_.trigPathSummaries[i].moduleInPathSummaries.size(); j++)
	{
	  //reset module summaries
	  trp_.trigPathSummaries[i].moduleInPathSummaries[j].timesVisited = 0;
	  trp_.trigPathSummaries[i].moduleInPathSummaries[j].timesPassed = 0;
	  trp_.trigPathSummaries[i].moduleInPathSummaries[j].timesFailed = 0;
	  trp_.trigPathSummaries[i].moduleInPathSummaries[j].timesExcept = 0;
	}
    }
  for(unsigned int i = 0; i < trp_.endPathSummaries.size(); i++)
    {
      trp_.endPathSummaries[i].timesRun    = 0;
      trp_.endPathSummaries[i].timesPassed = 0;
      trp_.endPathSummaries[i].timesFailed = 0;
      trp_.endPathSummaries[i].timesExcept = 0;
      for(unsigned int j = 0; j<trp_.endPathSummaries[i].moduleInPathSummaries.size(); j++)
	{
	  trp_.endPathSummaries[i].moduleInPathSummaries[j].timesVisited = 0;
	  trp_.endPathSummaries[i].moduleInPathSummaries[j].timesPassed = 0;
	  trp_.endPathSummaries[i].moduleInPathSummaries[j].timesFailed = 0;
	  trp_.endPathSummaries[i].moduleInPathSummaries[j].timesExcept = 0;
	}
    }

}


}//end namespace fuep
}//end namespace evf
