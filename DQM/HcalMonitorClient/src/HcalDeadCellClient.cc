#include <DQM/HcalMonitorClient/interface/HcalDeadCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

using namespace cms;
using namespace edm;
using namespace std;

HcalDeadCellClient::HcalDeadCellClient(const ParameterSet& ps, DaqMonitorBEInterface* dbe){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  dbe_ = dbe;

  if (verbose_)
    cout <<"Initializing HcalDeadCellClient from ParameterSet"<<endl;

  hbhists.type=0;
  hehists.type=1;
  hohists.type=2;
  hfhists.type=3;
 
  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "Hcal/");
  
  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
  
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB") subDetsOn_[0] = true;
    else if(subdets[i]=="HE") subDetsOn_[1] = true;
    else if(subdets[i]=="HO") subDetsOn_[2] = true;
    else if(subdets[i]=="HF") subDetsOn_[3] = true;
  }
}

HcalDeadCellClient::HcalDeadCellClient(){
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  dbe_ = 0;

  if (verbose_)
    cout <<"Initializing HcalDeadCellClient *without* ParameterSet"<<endl;
  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);

  hbhists.type=0;
  hehists.type=1;
  hohists.type=2;
  hfhists.type=3;
  // verbosity switch
  verbose_ = false;
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
}

HcalDeadCellClient::~HcalDeadCellClient(){

  this->cleanup();

}

void HcalDeadCellClient::beginJob(void){
  
  if ( verbose_ ) cout << "HcalDeadCellClient: beginJob" << endl;
  
  ievt_ = 0;
  jevt_ = 0;

  this->setup();
  this->resetAllME();
  return;
}

void HcalDeadCellClient::beginRun(void){

  if ( verbose_ ) cout << "HcalDeadCellClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalDeadCellClient::endJob(void) {

  if ( verbose_ ) cout << "HcalDeadCellClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalDeadCellClient::endRun(void) {

  if ( verbose_ ) cout << "HcalDeadCellClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();  
  return;
}

void HcalDeadCellClient::setup(void) {
  
  return;
}

void HcalDeadCellClient::cleanup(void) {

  if (verbose_)
    cout <<"HcalDeadCellClient::cleanup"<<endl;
  if ( cloneME_ ) 
    {
      deleteHists(hbhists);
      deleteHists(hehists);
      deleteHists(hohists);
      deleteHists(hfhists);
    }    

  clearHists(hbhists);
  clearHists(hehists);
  clearHists(hohists);
  clearHists(hfhists);

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalDeadCellClient::errorOutput(){
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
  cout <<"DeadCell Task: "<<dqmReportMapErr_.size()<<" errors, "<<dqmReportMapWarn_.size()<<" warnings, "<<dqmReportMapOther_.size()<<" others"<<endl;

  return;
}

void HcalDeadCellClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalDeadCellClient::report(){

  if ( verbose_ ) cout << "HcalDeadCellClient: report" << endl;
  //  this->setup();  
  
  char name[256];
  sprintf(name, "%sHcal/DeadCellMonitor/DeadCell Task Event Number",process_.c_str());
  MonitorElement* me = 0;
  if(dbe_) me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( verbose_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();

  return;
}

void HcalDeadCellClient::analyze(void){

  jevt_++;
  int updates = 0;
  //  if(dbe_) dbe_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalDeadCellClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalDeadCellClient::clearHists(DeadCellHists& hist)
{
  cout <<"Clearing HcalDeadCell histograms for HCAL type: "<<hist.type<<endl;

  hist.deadADC_OccMap=0;
  hist.deadADC_Eta=0;
  hist.badCAPID_OccMap=0;
  hist.badCAPID_Eta=0;
  hist.ADCdist=0;
  hist.NADACoolCellMap=0;
  hist.digiCheck=0;
  hist.cellCheck=0;
  hist.AbovePed=0;
  hist.CoolCellBelowPed=0;
  hist.DeadCap.clear();

  return;
}

void HcalDeadCellClient::deleteHists(DeadCellHists& hist)
{
  if (hist.deadADC_OccMap) delete hist.deadADC_OccMap;
  if (hist.deadADC_Eta) delete hist.deadADC_Eta;
  if (hist.badCAPID_OccMap) delete hist.badCAPID_OccMap;
  if (hist.badCAPID_Eta) delete hist.badCAPID_Eta;
  if (hist.ADCdist) delete hist.ADCdist;
  if (hist.NADACoolCellMap) delete hist.NADACoolCellMap;
  if (hist.digiCheck) delete hist.digiCheck;
  if (hist.cellCheck) delete hist.cellCheck;
  if (hist.AbovePed) delete hist.AbovePed;
  if (hist.CoolCellBelowPed) delete hist.CoolCellBelowPed;
  hist.DeadCap.clear();
  return;
}



void HcalDeadCellClient::getHistograms(){
  if(!dbe_) return;

  if(subDetsOn_[0]) getSubDetHistograms(hbhists);
  if(subDetsOn_[1]) getSubDetHistograms(hehists);
  if(subDetsOn_[2]) getSubDetHistograms(hohists);
  if(subDetsOn_[3]) getSubDetHistograms(hfhists);
  
  return;
}

void HcalDeadCellClient::getSubDetHistograms(DeadCellHists& hist)
{
  if (verbose_)
    cout <<"Getting subdetector histograms for subdetector "<<hist.type<<endl;
  
  char name[150];
  string type = "HB";
  if(hist.type==1) type = "HE"; 
  else if(hist.type==2) type = "HO"; 
  else if(hist.type==3) type = "HF"; 
  
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCOccupancyMap",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.deadADC_OccMap = getHisto2(name, process_, dbe_,verbose_,cloneME_); 
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCEta",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.deadADC_Eta = getHisto(name, process_, dbe_,verbose_,cloneME_);      
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDOccupancyMap",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.badCAPID_OccMap = getHisto2(name, process_, dbe_,verbose_,cloneME_); 
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDEta",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.badCAPID_Eta = getHisto(name, process_, dbe_,verbose_,cloneME_);      
 
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.ADCdist = getHisto(name, process_, dbe_,verbose_,cloneME_);  
  
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.NADACoolCellMap=getHisto2(name, process_, dbe_,verbose_,cloneME_);

  sprintf(name,"DeadCellMonitor/%s/%s_digiCheck",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.digiCheck=getHisto2(name, process_, dbe_,verbose_,cloneME_);

  sprintf(name,"DeadCellMonitor/%s/%s_cellCheck",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.cellCheck=getHisto2(name, process_, dbe_,verbose_,cloneME_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.AbovePed=getHisto2(name, process_, dbe_,verbose_,cloneME_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.CoolCellBelowPed=getHisto2(name, process_, dbe_,verbose_,cloneME_);
  
  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/%s_DeadCap%i",type.c_str(),type.c_str(),i);
      //cout <<name<<endl;
      hist.DeadCap.push_back(getHisto2(name, process_, dbe_,verbose_,cloneME_));
    }
  return;
}

  
void HcalDeadCellClient::getSubDetHistogramsFromFile(DeadCellHists& hist, TFile* infile)
{
  if (verbose_)
    cout <<"Getting subdetector histograms from file for subdetector "<<hist.type<<endl;
  
  char name[150];
  string type = "HB";
  if(hist.type==1) type = "HE"; 
  else if(hist.type==2) type = "HO"; 
  else if(hist.type==3) type = "HF"; 
  
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCOccupancyMap",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.deadADC_OccMap = (TH2F*)infile->Get(name); 
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCEta",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.deadADC_Eta = (TH1F*)infile->Get(name);      
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDOccupancyMap",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.badCAPID_OccMap = (TH2F*)infile->Get(name); 
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDEta",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.badCAPID_Eta = (TH1F*)infile->Get(name);      
 
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.ADCdist = (TH1F*)infile->Get(name);  
  
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.NADACoolCellMap=(TH2F*)infile->Get(name);

  sprintf(name,"DeadCellMonitor/%s/%s_digiCheck",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.digiCheck=(TH2F*)infile->Get(name);

  sprintf(name,"DeadCellMonitor/%s/%s_cellCheck",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.cellCheck=(TH2F*)infile->Get(name);
  
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.AbovePed=(TH2F*)infile->Get(name);
  
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  //cout <<name<<endl;
  hist.CoolCellBelowPed=(TH2F*)infile->Get(name);
  
  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/%s_DeadCap%i",type.c_str(),type.c_str(),i);
      //cout <<name<<endl;
      hist.DeadCap.push_back((TH2F*)infile->Get(name));
    }
  return;

}

void HcalDeadCellClient::resetSubDetHistograms(DeadCellHists& hist)
{
  if (verbose_)
    cout <<"Resetting subdetector histograms for subdetector "<<hist.type<<endl;
  
  char name[150];
  string type = "HB";
  if(hist.type==1) type = "HE"; 
  else if(hist.type==2) type = "HO"; 
  else if(hist.type==3) type = "HF"; 
  
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCOccupancyMap",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_deadADCEta",type.c_str(),type.c_str());
  resetME(name,dbe_);
  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDOccupancyMap",type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"DeadCellMonitor/%s/%s_noADCIDEta",type.c_str(),type.c_str());
  resetME(name,dbe_);
 
  sprintf(name,"DeadCellMonitor/%s/%s_ADCdist",type.c_str(),type.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_NADA_CoolCellMap",type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"DeadCellMonitor/%s/%s_digiCheck",type.c_str(),type.c_str());
  resetME(name,dbe_);

  sprintf(name,"DeadCellMonitor/%s/%s_cellCheck",type.c_str(),type.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_abovePed",type.c_str(),type.c_str());
  resetME(name,dbe_);
  
  sprintf(name,"DeadCellMonitor/%s/%s_CoolCell_belowPed",type.c_str(),type.c_str());
  resetME(name,dbe_);
  
  for (int i=0;i<4;i++)
    {
      sprintf(name,"DeadCellMonitor/%s/%s_DeadCap%i",type.c_str(),type.c_str(),i);
      resetME(name,dbe_); 
      //cout <<name<<endl;
    }
  return;

}


void HcalDeadCellClient::resetAllME(){
  if(!dbe_) return;
  if (subDetsOn_[0]) resetSubDetHistograms(hbhists);
  if (subDetsOn_[1]) resetSubDetHistograms(hehists);
  if (subDetsOn_[2]) resetSubDetHistograms(hohists);
  if (subDetsOn_[3]) resetSubDetHistograms(hfhists);

  return;
}

void HcalDeadCellClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalDeadCellClient html output ..." << endl;
  string client = "DeadCellMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal DeadCell Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal DeadCells</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;

  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"DeadCellMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"DeadCellMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"DeadCellMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Dead Cell Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;  
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  //htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;

  htmlSubDetOutput(hbhists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hehists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hohists,runNo,htmlDir,htmlName);
  htmlSubDetOutput(hfhists,runNo,htmlDir,htmlName);

  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  return;
}

void HcalDeadCellClient::htmlSubDetOutput(DeadCellHists& hist, int runNo,
					  string htmlDir,
					  string htmlName)
 {
   if(!subDetsOn_[hist.type]) return;
    
    string type = "HB";
    if(hist.type==1) type = "HE"; 
    if(hist.type==2) type = "HF"; 
    if(hist.type==3) type = "HO"; 

    htmlFile << "<tr align=\"left\">" << endl;
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(runNo,hist.deadADC_OccMap,"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML(runNo,hist.deadADC_Eta,"iEta","Evts", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(runNo,hist.digiCheck,"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(runNo,hist.cellCheck,"iEta","iPhi", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(runNo,hist.NADACoolCellMap,"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(runNo,hist.CoolCellBelowPed,"iEta","iPhi", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(runNo,hist.DeadCap[0],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(runNo,hist.DeadCap[1],"iEta","iPhi", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;	
    histoHTML2(runNo,hist.DeadCap[2],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(runNo,hist.DeadCap[3],"iEta","iPhi", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
}

void HcalDeadCellClient::createTests(){
  //  char meTitle[250], name[250];    
  //  vector<string> params;
  
  if(verbose_) cout <<"Creating DeadCell tests..."<<endl;
  createSubDetTests(hbhists);
  createSubDetTests(hehists);
  createSubDetTests(hohists);
  createSubDetTests(hfhists);
    
  return;
}

void HcalDeadCellClient::createSubDetTests(DeadCellHists& hist)
{
  if(!subDetsOn_[hist.type]) return;
  if (verbose_) 
    cout <<"Running HcalDeadCellClient::createSubDetTests for subdetector: "<<hist.type<<endl;
  char meTitle[250], name[250];
  vector<string> params;

  string type="HB";
  if(hist.type==1)
    type="HE";
  else if (hist.type==2)
    type="HO";
  else if (hist.type==3)
    type="HF";


  // Check for dead ADCs
  sprintf(meTitle,"%sHcal/DeadCellMonitor/%s/%s_deadADCOccupancyMap",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s Dead ADC Map",type.c_str()); 
  if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	}
      else
	if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    }

  // Check NADA cool cells
  sprintf(meTitle,"%sHcal/DeadCellMonitor/%s/%s_NADA_CoolCellMap",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Cool Cell Map",type.c_str()); 
  if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	}
      else
	if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    }
  
  // Check for cells consistently below pedestal+nsigma
  sprintf(meTitle,"%sHcal/DeadCellMonitor/%s/%s_HE_CoolCell_belowPed",process_.c_str(),type.c_str(), type.c_str());
  sprintf(name,"%s NADA Cool Cell Map",type.c_str()); 
  if (verbose_) cout <<"Checking for histogram named: "<<name<<endl;
  if(dqmQtests_.find(name)==dqmQtests_.end())
    {
      if (verbose_) cout <<"Didn't find histogram; search for title: "<<meTitle<<endl;
      MonitorElement* me = dbe_->get(meTitle);
      if (me)
	{
	  if (verbose_) cout <<"Got histogram with title "<<meTitle<<"\nChecking for content"<<endl;
	  dqmQtests_[name]=meTitle;
	  params.clear();
	  params.push_back((string)meTitle);
	  params.push_back((string)name);
	  createH2ContentTest(dbe_,params);
	}
      else
	if (verbose_) cout <<"Couldn't find histogram with title: "<<meTitle<<endl;
    }
  return;
}


void HcalDeadCellClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/DeadCellMonitor/DeadCell Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  getSubDetHistogramsFromFile(hbhists,infile);
  getSubDetHistogramsFromFile(hehists,infile);
  getSubDetHistogramsFromFile(hohists,infile);
  getSubDetHistogramsFromFile(hfhists,infile);

  return;
}


