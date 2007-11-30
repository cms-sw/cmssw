#include <DQM/HcalMonitorClient/interface/HcalBaseClient.h>

HcalBaseClient::HcalBaseClient(const ParameterSet& ps, DaqMonitorBEInterface* dbe){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  dbe_ = dbe;
  ievt_=0; jevt_=0;
  
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);
  
  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "Hcal/");

  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
  
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB") subDetsOn_[0] = true;
    else if(subdets[i]=="HE") subDetsOn_[1] = true;
    else if(subdets[i]=="HF") subDetsOn_[2] = true;
    else if(subdets[i]=="HO") subDetsOn_[3] = true;
  }
}

HcalBaseClient::HcalBaseClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  debug_ =false;
  dbe_ = 0;
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
}

HcalBaseClient::~HcalBaseClient(){}

void HcalBaseClient::beginJob(void){

  if ( debug_ ) cout << "HcalBaseClient: beginJob" << endl;
  ievt_ = 0; jevt_ = 0;
  return;
}

void HcalBaseClient::beginRun(void){

  if ( debug_ ) cout << "HcalBaseClient: beginRun" << endl;
  jevt_ = 0;
  return;
}

void HcalBaseClient::endJob(void) {
  if ( debug_ ) cout << "HcalBaseClient: endJob, ievt = " << ievt_ << endl;
  return;
}

void HcalBaseClient::endRun(void) {
  if ( debug_ ) cout << "HcalBaseClient: endRun, jevt = " << jevt_ << endl;
  return;
}

void HcalBaseClient::errorOutput(){
  
  if(!dbe_) return;

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); testsMap!=dqmQtests_.end();testsMap++){
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
  return;
}

void HcalBaseClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

  this->errorOutput();
  outE.clear(); outW.clear(); outO.clear();

  for(map<string, vector<QReport*> >::iterator i=dqmReportMapErr_.begin(); i!=dqmReportMapErr_.end(); i++){
    outE[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapWarn_.begin(); i!=dqmReportMapWarn_.end(); i++){
    outW[i->first] = i->second;
  }
  for(map<string, vector<QReport*> >::iterator i=dqmReportMapOther_.begin(); i!=dqmReportMapOther_.end(); i++){
    outO[i->first] = i->second;
  }

  return;
}
