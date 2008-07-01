#include <DQM/HcalMonitorClient/interface/HcalBaseClient.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalBaseClient::HcalBaseClient(){
  dbe_ =NULL;
  clientName_ = "GenericHcalClient";
}

HcalBaseClient::~HcalBaseClient(){}

void HcalBaseClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)
{
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  dbe_ = dbe;
  ievt_=0; jevt_=0;
  clientName_ = clientName;
  
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);
  if(debug_) cout << clientName_ <<" debugging switch is on"<<endl;
  
  // timing switch
  showTiming_ = ps.getUntrackedParameter<bool>("showTiming",false); 

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "Hcal/");
  
  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++)
    {
      subDetsOn_[i] = false;
    }

  for(unsigned int i=0; i<subdets.size(); i++)
    {
      if(subdets[i]=="HB") subDetsOn_[0] = true;
      else if(subdets[i]=="HE") subDetsOn_[1] = true;
      else if(subdets[i]=="HF") subDetsOn_[2] = true;
      else if(subdets[i]=="HO") subDetsOn_[3] = true;
    }
  
  return; 
} // void HcalBaseClient::init(const ParameterSet& ps, DQMStore* dbe, string clientName)


void HcalBaseClient::errorOutput(){
  
  if(!dbe_) return;

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); 
       testsMap!=dqmQtests_.end();testsMap++){
    string testName = testsMap->first;
    string meName = testsMap->second;
    MonitorElement* me = dbe_->get(meName);
    if(me){
      if (me->hasError()){
	vector<QReport*> report =  me->getQErrors();
	dqmReportMapErr_[meName] = report;
      }
      if (me->hasWarning()){
	vector<QReport*> report =  me->getQWarnings();
	dqmReportMapWarn_[meName] = report;
      }
      if(me->hasOtherReport()){
	vector<QReport*> report= me->getQOthers();
	dqmReportMapOther_[meName] = report;
      }
    }
  }

  cout << clientName_ << " Error Report: "<< dqmQtests_.size() << " tests, "<<dqmReportMapErr_.size() << " errors, " <<dqmReportMapWarn_.size() << " warnings, "<< dqmReportMapOther_.size() << " others" << endl;

  return;
}

void HcalBaseClient::getTestResults(int& totalTests, 
				    map<string, vector<QReport*> >& outE, 
				    map<string, vector<QReport*> >& outW, 
				    map<string, vector<QReport*> >& outO){
  this->errorOutput();
  //  outE.clear(); outW.clear(); outO.clear();

  for(map<string, vector<QReport*> >::iterator i=dqmReportMapErr_.begin(); i!=dqmReportMapErr_.end(); i++){
    outE[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapWarn_.begin(); i!=dqmReportMapWarn_.end(); i++){
    outW[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapOther_.begin(); i!=dqmReportMapOther_.end(); i++){
    outO[i->first] = i->second;
  }

  totalTests += dqmQtests_.size();

  return;
}
