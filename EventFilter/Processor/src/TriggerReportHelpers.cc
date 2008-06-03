#include "TriggerReportHelpers.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "xdata/TableIterator.h"


#include <iostream>

namespace evf{
  namespace fuep{
const std::string TriggerReportHelpers::columns[6] = {"pathName","l1Pass","psPass","pAccept","pExcept","pReject"};
void TriggerReportHelpers::triggerReportToTable(edm::TriggerReport &tr, unsigned int ls, bool lumiComplete)
{
  std::cout <<"Calling triggerReportToTable with ls " << ls << " and lc " << lumiComplete << std::endl;
  lumiSectionIndex_ = ls;  
  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    if(l1pos_[i]>=0) {
      std::cout << "before " << l1pre_[i] << " " << pl1pre_[i] << std::endl;
      l1pre_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed 
	+ (lumiComplete ? - pl1pre_[i]  : l1pre_[i].value_);
      pl1pre_[i] = tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed;
      std::cout << "after " << l1pre_[i] << " " <<  tr.trigPathSummaries[i].moduleInPathSummaries[l1pos_[i]].timesPassed
		<< " " << pl1pre_[i] << std::endl;
    }
    else {
      std::cout << "before " << l1pre_[i] << " " << pl1pre_[i] << std::endl;
      l1pre_[i] = tr.trigPathSummaries[i].timesRun 
	+ (lumiComplete ? - pl1pre_[i] : l1pre_[i].value_);
      pl1pre_[i] = tr.trigPathSummaries[i].timesRun;
      std::cout << "after " << l1pre_[i] << " " <<  tr.trigPathSummaries[i].timesRun
		<< " " << pl1pre_[i] << std::endl;
    }
    triggerReportAsTable_.setValueAt(i,columns[1],l1pre_[i]);
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
    triggerReportAsTable_.setValueAt(i,columns[2],ps_[i]);
    accept_[i] = tr.trigPathSummaries[i].timesPassed 
	+ (lumiComplete ? - paccept_[i] : accept_[i].value_);
    paccept_[i] = tr.trigPathSummaries[i].timesPassed; 
    triggerReportAsTable_.setValueAt(i,columns[3], accept_[i]);
    except_[i] = tr.trigPathSummaries[i].timesExcept 
	+ (lumiComplete ? - pexcept_[i] : except_[i].value_);
    pexcept_[i] = tr.trigPathSummaries[i].timesExcept;
    triggerReportAsTable_.setValueAt(i,columns[4], except_[i]);
    failed_[i] = tr.trigPathSummaries[i].timesFailed 
	+ (lumiComplete ? - pfailed_[i] : failed_[i].value_);
    pfailed_[i] = tr.trigPathSummaries[i].timesFailed;
    triggerReportAsTable_.setValueAt(i,columns[5], failed_[i]);
  }
}

void TriggerReportHelpers::formatReportTable(edm::TriggerReport &tr, std::vector<edm::ModuleDescription const*>& descs)  
{
  if(tableFormatted_) return;
  tableFormatted_ = true;
  paths_.resize(tr.trigPathSummaries.size());
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
  triggerReportAsTable_.addColumn(columns[0],"string");
  triggerReportAsTable_.addColumn(columns[1],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[2],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[3],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[4],"unsigned int 32");
  triggerReportAsTable_.addColumn(columns[5],"unsigned int 32");
  for(unsigned int i=0; i<tr.trigPathSummaries.size(); i++) {
    xdata::Table::iterator it = triggerReportAsTable_.append();
    paths_[i] = tr.trigPathSummaries[i].name;
    it->setField(columns[0],paths_[i]);
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
}
}
