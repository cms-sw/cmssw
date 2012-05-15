#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Modules/interface/ShmOutputModuleRegistry.h"

#include "TriggerReportHelpers.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "xdata/TableIterator.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <map>

#include "boost/tokenizer.hpp"

namespace evf{
  namespace fuep{
const std::string TriggerReportHelpers::columns[5] = {"l1Pass","psPass","pAccept","pExcept","pReject"};
void TriggerReportHelpers::triggerReportUpdate(edm::TriggerReport &tr, unsigned int ls, unsigned int ps, bool lumiComplete)
{
  if(adjustLsIndex_)
    {
      lumiSectionIndex_=ls;
      adjustLsIndex_ = false;
    }
  else
    lumiSectionIndex_++;  
  if(lumiSectionIndex_ != ls)
    std::cout << getpid() << " WARNING: ls index mismatch " << ls << " should be " << lumiSectionIndex_ << std::endl;
  prescaleIndex_ = ps;
}
    
void TriggerReportHelpers::packedTriggerReportToTable()
{

  TriggerReportStatic *trs = getPackedTriggerReportAsStruct();

  eventsProcessed_.value_ = trs->eventSummary.totalEvents;
  eventsAccepted_.value_  = trs->eventSummary.totalEventsPassed;
  for(int i=0; i<trs->trigPathsInMenu; i++) {
    pl1pre_[i] += (l1pre_[i] =trs->trigPathSummaries[i].timesPassedL1);
    triggerReportAsTable_.setValueAt(i,columns[0],l1pre_[i]);
    triggerReportAsTableWithNames_.setValueAt(i,columns[0],l1pre_[i]);
    pps_[i] += (ps_[i] = trs->trigPathSummaries[i].timesPassedPs);
    triggerReportAsTable_.setValueAt(i,columns[1],ps_[i]);
    triggerReportAsTableWithNames_.setValueAt(i,columns[1],ps_[i]);
    paccept_[i] += (accept_[i] = trs->trigPathSummaries[i].timesPassed);
    triggerReportAsTable_.setValueAt(i,columns[2], accept_[i]);
    triggerReportAsTableWithNames_.setValueAt(i,columns[2], accept_[i]);
    pexcept_[i] += (except_[i] = trs->trigPathSummaries[i].timesExcept);
    triggerReportAsTable_.setValueAt(i,columns[3], except_[i]);
    triggerReportAsTableWithNames_.setValueAt(i,columns[3], except_[i]);
    pfailed_[i] += (failed_[i] = trs->trigPathSummaries[i].timesFailed);
    triggerReportAsTable_.setValueAt(i,columns[4], failed_[i]);
    triggerReportAsTableWithNames_.setValueAt(i,columns[4], failed_[i]);
  }
  for(int i=0; i<trs->endPathsInMenu; i++) {    
    int j = i+trs->trigPathsInMenu;
    pl1pre_[j] += (l1pre_[j] = trs->endPathSummaries[i].timesPassedL1);
    triggerReportAsTable_.setValueAt(j,columns[0],l1pre_[j]);
    triggerReportAsTableWithNames_.setValueAt(j,columns[0],l1pre_[j]);
    pps_[j] += (ps_[j] = trs->endPathSummaries[i].timesPassedPs);
    triggerReportAsTable_.setValueAt(j,columns[1],ps_[j]);
    triggerReportAsTableWithNames_.setValueAt(j,columns[1],ps_[j]);
    paccept_[j] += (accept_[j] = trs->endPathSummaries[i].timesPassed);
    triggerReportAsTable_.setValueAt(j,columns[2], accept_[j]);
    triggerReportAsTableWithNames_.setValueAt(j,columns[2], accept_[j]);
    pexcept_[j] += (except_[j] = trs->endPathSummaries[i].timesExcept);
    triggerReportAsTable_.setValueAt(j,columns[3], except_[j]);
    triggerReportAsTableWithNames_.setValueAt(j,columns[3], except_[j]);
    pfailed_[j] += (failed_[j] = trs->endPathSummaries[i].timesFailed);
    triggerReportAsTable_.setValueAt(j,columns[4], failed_[j]);
    triggerReportAsTableWithNames_.setValueAt(j,columns[4], failed_[j]);
  }
}

void TriggerReportHelpers::fillPathIndexTable(std::string &pathstring)
{
  unsigned int i = 0;
  if(pathstring == ""){
    for(; i<paths_.size(); i++) {
      xdata::Table::iterator it = triggerReportAsTableWithNames_.append();
      it->setField("pathIndex", pathIndexMap_[paths_[i]]=i);
    }
  }
  else{
    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char> > tokens(pathstring, sep);
    for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
	 tok_iter != tokens.end(); ++tok_iter){
      unsigned int index = 0;
      std::string name;
      std::string::size_type pos = tok_iter->find("=");
      if(pos!=std::string::npos){
	name=tok_iter->substr(0,pos);
	index = atoi(tok_iter->substr(pos+1).c_str());
	pathIndexMap_[name]=index;
      }
    }
    for(; i<paths_.size(); i++) {
      if(pathIndexMap_.find(paths_[i])==pathIndexMap_.end())
	pathIndexMap_[paths_[i]] = i;
      xdata::Table::iterator it = triggerReportAsTableWithNames_.append();
      it->setField("pathIndex",pathIndexMap_[paths_[i]]);
    }
  }
}

void TriggerReportHelpers::formatReportTable(edm::TriggerReport &tr, 
					     std::vector<edm::ModuleDescription const*>& descs,
					     std::string &pathIndexTable,
					     bool noNukeLegenda)  
{

  if(tableFormatted_) return;
  std::ostringstream ost;
  trp_ = tr;
  lumiSectionIndex_ = 0;
  resetTriggerReport();
  TriggerReportStatic *trp = (TriggerReportStatic *)cache_->mtext;

  tableFormatted_ = true;
  paths_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size());
  //adjust number of trigger- and end-paths in static structure
  trp->trigPathsInMenu = tr.trigPathSummaries.size();
  trp->endPathsInMenu = tr.endPathSummaries.size();
  ///

  l1pos_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),-1);
  pspos_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),-1);
  outname_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size());
  l1pre_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  ps_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  accept_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  except_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  failed_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  pl1pre_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  pps_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  paccept_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  pexcept_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  pfailed_.resize(tr.trigPathSummaries.size()+tr.endPathSummaries.size(),0);
  triggerReportAsTable_.clear();
  triggerReportAsTableWithNames_.clear();
  triggerReportAsTable_.addColumn(columns[0],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[1],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[2],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[3],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[4],"unsigned int 32");
  triggerReportAsTableWithNames_ = triggerReportAsTable_;
  triggerReportAsTableWithNames_.addColumn("pathIndex","unsigned int 32");

  unsigned int i=0;
  for(; i<tr.trigPathSummaries.size(); i++) {
    triggerReportAsTable_.append();
    paths_[i] = tr.trigPathSummaries[i].name;

    ost << i << "=" << paths_[i] << ", ";

    // reset the l1 and ps positions to pick up modifications of the menu
    // that result in paths being displaced up and down
    l1pos_[i] = -1;
    pspos_[i] = -1;

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
  for(; i<tr.endPathSummaries.size()+tr.trigPathSummaries.size(); i++) {
    triggerReportAsTable_.append();
    paths_[i] = tr.endPathSummaries[i-tr.trigPathSummaries.size()].name;
    ost << i << "=" << paths_[i] << ", ";
    // reset the l1 and ps positions to pick up modifications of the menu
    // that result in paths being displaced up and down
    l1pos_[i] = -1;
    pspos_[i] = -1;
    outname_[i] = "";

    for(unsigned int j=0;
	j<tr.endPathSummaries[i-tr.trigPathSummaries.size()].moduleInPathSummaries.size();
	j++) {
      std::string label = tr.endPathSummaries[i-tr.trigPathSummaries.size()].moduleInPathSummaries[j].moduleLabel;
      for(unsigned int k = 0; k < descs.size(); k++)
	{
	  if(descs[k]->moduleLabel() == label) 
	    {
	      if(descs[k]->moduleName() == "TriggerResultsFilter") pspos_[i] = j;
	      //	      if(descs[k]->moduleName() == "HLTPrescaler") l1pos_[i] = j;
	      if(descs[k]->moduleName() == "ShmStreamConsumer") 
		outname_[i] = descs[k]->moduleLabel();
	    }
	}

    }
  }
  fillPathIndexTable(pathIndexTable);
  if(noNukeLegenda) pathLegenda_ = ost.str().c_str();

}

void TriggerReportHelpers::printReportTable()
{
  std::cout << "report table for LS #" << lumiSectionIndex_ << std::endl;
  triggerReportAsTableWithNames_.writeTo(std::cout);
  std::cout << std::endl;
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

void TriggerReportHelpers::packTriggerReport(edm::TriggerReport &tr,
					     ShmOutputModuleRegistry *sor)
{
  TriggerReportStatic *trp = (TriggerReportStatic *)cache_->mtext;
  trp->lumiSection = lumiSectionIndex_;
  trp->prescaleIndex = prescaleIndex_;
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

  for(int i = 0; i < trp->trigPathsInMenu; i++)
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

  for(int i = 0; i < trp->endPathsInMenu; i++)
    {
      unsigned int j = i + trp->trigPathsInMenu;
      evf::OutputModule *o = sor->get(outname_[j]);
      if(!o) {
	//	sor->dumpRegistry();
	continue;
      }
      trp->endPathSummaries[i].timesRun    = 
	tr.endPathSummaries[i].timesRun - trp_.endPathSummaries[i].timesRun;
      trp->endPathSummaries[i].timesPassed = 
        o->getCounts() - trp_.endPathSummaries[i].timesPassed;
      trp->endPathSummaries[i].timesFailed = 
	(tr.endPathSummaries[i].timesRun - o->getCounts()) 
	- trp_.endPathSummaries[i].timesFailed;
      trp->endPathSummaries[i].timesExcept = 
	tr.endPathSummaries[i].timesExcept - trp_.endPathSummaries[i].timesExcept;


      if(l1pos_[j]>=0) {
	trp->endPathSummaries[i].timesPassedL1 =
	  tr.endPathSummaries[i].moduleInPathSummaries[l1pos_[j]].timesPassed -
	  trp_.endPathSummaries[i].moduleInPathSummaries[l1pos_[j]].timesPassed;
      }
      else {
	trp->endPathSummaries[i].timesPassedL1 = trp->endPathSummaries[i].timesRun;
      }
      if(pspos_[j]>=0) {
	trp->endPathSummaries[i].timesPassedPs =
	  tr.endPathSummaries[i].moduleInPathSummaries[pspos_[j]].timesPassed -
	  trp_.endPathSummaries[i].moduleInPathSummaries[pspos_[j]].timesPassed;
      }
      else if(l1pos_[j]>=0) {
	trp->endPathSummaries[i].timesPassedPs =
	  tr.endPathSummaries[i].moduleInPathSummaries[l1pos_[j]].timesPassed -
	  trp_.endPathSummaries[i].moduleInPathSummaries[l1pos_[j]].timesPassed;
      }
      else {
	trp->endPathSummaries[i].timesPassedPs = trp->endPathSummaries[i].timesRun;
      }
    }
  trp_ = tr;
  for(int i = 0; i < trp->endPathsInMenu; i++)
    {
      evf::OutputModule *o = sor->get(outname_[i+trp->trigPathsInMenu]);
      if(!o) {
	//	sor->dumpRegistry();
	continue;
      }
      trp_.endPathSummaries[i].timesPassed = o->getCounts();
      trp_.endPathSummaries[i].timesFailed = tr.endPathSummaries[i].timesRun - o->getCounts();
    }
  
}


void TriggerReportHelpers::sumAndPackTriggerReport(MsgBuf &buf)
{

  TriggerReportStatic *trs = (TriggerReportStatic *)cache_->mtext;
  TriggerReportStatic *trp = (TriggerReportStatic *)buf->mtext;

  // add check for LS consistency
  if(trp->lumiSection != lumiSectionIndex_){
    std::cout << "WARNING: lumisection index mismatch from subprocess " << trp->lumiSection
	      << " should be " << lumiSectionIndex_ << " will be skipped" << std::endl;
    return;
  }
  //get total paths in the menu
  if(trs->trigPathsInMenu != trp->trigPathsInMenu) 
    {
      std::ostringstream ost;
      ost << "trig path summary inconsistency " 
	  << trs->trigPathsInMenu << " vs. " << trp->trigPathsInMenu;
      std::cout << ost.str() << std::endl;
      XCEPT_RAISE(evf::Exception,ost.str());
    }
  if(trs->endPathsInMenu != trp->endPathsInMenu)
    {
      std::ostringstream ost;
      ost << "trig endpath summary inconsistency " 
	  << trs->endPathsInMenu << " vs. " << trp->endPathsInMenu;
      std::cout << ost.str() << std::endl;
      XCEPT_RAISE(evf::Exception,ost.str());
    }
  funcs::addToReport(trs,trp,lumiSectionIndex_);
  
}  

void TriggerReportHelpers::resetPackedTriggerReport()
{

  TriggerReportStatic *trp = (TriggerReportStatic *)cache_->mtext;

  funcs::reset(trp);

  lumiSectionIndex_++;
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
std::string TriggerReportHelpers::findLabelOfModuleTypeInEndPath(edm::TriggerReport &tr, 
								 std::vector<edm::ModuleDescription const*>& descs,
								 unsigned int ind, 
								 std::string type)
{
  std::string retval;
  for(unsigned int j=0;
      j<tr.endPathSummaries[ind].moduleInPathSummaries.size();
      j++) {
    
    std::string label = tr.endPathSummaries[ind].moduleInPathSummaries[j].moduleLabel;
    for(unsigned int k = 0; k < descs.size(); k++)
      {
	if(descs[k]->moduleLabel() == label) 
	  {
	    if(descs[k]->moduleName() == type) {retval = label; break;}
	  }
      }
  }
  return retval;
}

}//end namespace fuep
}//end namespace evf
