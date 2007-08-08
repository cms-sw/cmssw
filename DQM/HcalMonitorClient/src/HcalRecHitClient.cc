#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalRecHitClient::HcalRecHitClient(const ParameterSet& ps, MonitorUserInterface* mui){

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<4; i++){
    occ_[i]=0;
    energy_[i]=0;
    energyT_[i]=0;
    time_[i]=0;
    tot_occ_[i]=0;
  }
  tot_energy_=0;

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  beamE_thresh_ = ps.getUntrackedParameter<double>("beamEnergyMaxThresh", 0.9);
  cout << "Beam energy maximum threshold set to " << beamE_thresh_ << endl;

  beamE_width_ = ps.getUntrackedParameter<double>("beamEnergyWidth", 0.1);
  cout << "Beam energy reco width set to " << beamE_thresh_ << endl;

  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor");

  vector<string> subdets = ps.getUntrackedParameter<vector<string> >("subDetsOn");
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
  
  for(unsigned int i=0; i<subdets.size(); i++){
    if(subdets[i]=="HB") subDetsOn_[0] = true;
    else if(subdets[i]=="HE") subDetsOn_[1] = true;
    else if(subdets[i]=="HF") subDetsOn_[2] = true;
    else if(subdets[i]=="HO") subDetsOn_[3] = true;
  }
}

HcalRecHitClient::HcalRecHitClient(){

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = 0;
  for(int i=0; i<4; i++){
    occ_[i]=0;
    energy_[i]=0;
    energyT_[i]=0;
    time_[i]=0;
    tot_occ_[i]=0;
  }
  tot_energy_=0;

  // verbosity switch
  verbose_ = false;
  for(int i=0; i<4; i++) subDetsOn_[i] = false;
}

HcalRecHitClient::~HcalRecHitClient(){

  this->cleanup();

}

void HcalRecHitClient::beginJob(void){

  if ( verbose_ ) cout << "HcalRecHitClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->setup();
  this->subscribe();
  this->resetAllME();
  return;
}

void HcalRecHitClient::beginRun(void){

  if ( verbose_ ) cout << "HcalRecHitClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetAllME();
  return;
}

void HcalRecHitClient::endJob(void) {

  if ( verbose_ ) cout << "HcalRecHitClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalRecHitClient::endRun(void) {

  if ( verbose_ ) cout << "HcalRecHitClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
}

void HcalRecHitClient::setup(void) {
  return;
}

void HcalRecHitClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<4; i++){
      if(occ_[i]) delete occ_[i];
      if(energy_[i]) delete energy_[i];
      if(energyT_[i]) delete energyT_[i];
      if(time_[i]) delete time_[i];
      if(tot_occ_[i]) delete tot_occ_[i];
    }    

    if(tot_energy_) delete tot_energy_;
  }
  
  for(int i=0; i<4; i++){
    occ_[i]=0; energy_[i]=0;
    energyT_[i]=0; time_[i]=0;
    tot_occ_[i]=0;
  }
  tot_energy_=0;
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
}

void HcalRecHitClient::subscribe(void){

  if ( verbose_ ) cout << "HcalRecHitClient: subscribe" << endl;
  if(mui_){
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HB/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HE/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HF/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HO/*");
  }
  return;
}

void HcalRecHitClient::subscribeNew(void){
  if(mui_){
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HB/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HE/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HF/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HO/*");
  }
  return;
}

void HcalRecHitClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalRecHitClient: unsubscribe" << endl;
  if(mui_){
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HB/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HE/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HF/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HO/*");
  }
  return;
}

void HcalRecHitClient::errorOutput(){
  if(!mui_) return;

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  
  for (map<string, string>::iterator testsMap=dqmQtests_.begin(); testsMap!=dqmQtests_.end();testsMap++){
    string testName = testsMap->first;
    string meName = testsMap->second;
    MonitorElement* me = mui_->get(meName);
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
  printf("RecHit Task: %d errors, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());

  return;
}

void HcalRecHitClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalRecHitClient::report(){
  if(!mui_) return;
  if ( verbose_ ) cout << "HcalRecHitClient: report" << endl;
  this->setup();

  char name[256];
  sprintf(name, "%sHcalMonitor/RecHitMonitor/RecHit Event Number",process_.c_str());
  MonitorElement* me = mui_->get(name); 
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( verbose_ ) cout << "Found '" << name << "'" << endl;
  }

  getHistograms();

  return;
}

void HcalRecHitClient::analyze(void){

  jevt_++;
  int updates = 0;
  if(mui_) mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalRecHitClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalRecHitClient::getHistograms(){
  if(!mui_) return;
  char name[150];    
  for(int i=0; i<4; i++){
    sprintf(name,"RecHitMonitor/RecHit Depth %d Occupancy Map",i+1);
    tot_occ_[i] = getHisto2(name, process_, mui_, verbose_,cloneME_);
  }

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"RecHitMonitor/%s/%s RecHit Energies",type.c_str(),type.c_str());      
    energy_[i] = getHisto(name, process_,mui_,verbose_,cloneME_);
    
    sprintf(name,"RecHitMonitor/%s/%s RecHit Total Energy",type.c_str(),type.c_str());      
    energyT_[i] = getHisto(name, process_,mui_,verbose_,cloneME_);

    sprintf(name,"RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
    time_[i] = getHisto(name, process_,mui_,verbose_,cloneME_);

    sprintf(name,"RecHitMonitor/%s/%s RecHit Geo Occupancy Map",type.c_str(),type.c_str());
    occ_[i] = getHisto2(name, process_,mui_,verbose_,cloneME_);
  }

  
  sprintf(name,"RecHitMonitor/RecHit Total Energy");   
  tot_energy_ = getHisto(name, process_,mui_, verbose_,cloneME_);

  return;
}

void HcalRecHitClient::resetAllME(){
  if(!mui_) return;
  Char_t name[150];
  
  sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Total Energy",process_.c_str());
  resetME(name,mui_);
  for(int i=1; i<5; i++){
    sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Depth %d Occupancy Map",process_.c_str(),i);
    resetME(name,mui_);
    sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Depth %d Energy Map",process_.c_str(),i);
    resetME(name,mui_);
  }
  sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Eta Occupancy Map",process_.c_str());
  resetME(name,mui_);
  sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Phi Occupancy Map",process_.c_str());
  resetME(name,mui_);
  sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Eta Energy Map",process_.c_str());
  resetME(name,mui_);
  sprintf(name,"%sHcalMonitor/RecHitMonitor/RecHit Phi Energy Map",process_.c_str());
  resetME(name,mui_);

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 


    sprintf(name,"%sHcalMonitor/RecHitMonitor/%s/%s RecHit Geo Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,mui_);
    sprintf(name,"%sHcalMonitor/RecHitMonitor/%s/%s RecHit Energies",process_.c_str(),type.c_str(),type.c_str());      
    resetME(name,mui_);
    sprintf(name,"%sHcalMonitor/RecHitMonitor/%s/%s RecHit Energies - Low Region",process_.c_str(),type.c_str(),type.c_str());      
    resetME(name,mui_);
    sprintf(name,"%sHcalMonitor/RecHitMonitor/%s/%s RecHit Total Energy",process_.c_str(),type.c_str(),type.c_str());      
    resetME(name,mui_);
    sprintf(name,"%sHcalMonitor/RecHitMonitor/%s/%s RecHit Times",process_.c_str(),type.c_str(),type.c_str()); 
    resetME(name,mui_);     
  }


  return;
}


void HcalRecHitClient::htmlOutput(int runNo, string htmlDir, string htmlName){

  cout << "Preparing HcalRecHitClient html output ..." << endl;
  string client = "RecHitMonitor";
  htmlErrors(runNo,htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal RecHit Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal RecHits</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"RecHitMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"RecHitMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"RecHitMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal RecHit Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  if(subDetsOn_[0]) htmlFile << "<a href=\"#HB_Plots\">HB Plots </a></br>" << endl;
  if(subDetsOn_[1]) htmlFile << "<a href=\"#HE_Plots\">HE Plots </a></br>" << endl;
  if(subDetsOn_[2]) htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  if(subDetsOn_[3]) htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,tot_occ_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,tot_occ_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,tot_occ_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,tot_occ_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML(runNo,tot_energy_,"Total Energy (GeV)","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    htmlFile << "<tr align=\"left\">" << endl;

    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,occ_[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML(runNo,energyT_[i],"Total Energy (GeV)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,energy_[i],"RecHit Energy (GeV)","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,time_[i],"RecHit Time (nS)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;	
  }
  htmlFile << "</table>" << endl;
  htmlFile << "<br>" << endl;


  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

  return;
}

void HcalRecHitClient::createTests(){
  if(!mui_) return;
  //  char meTitle[250], name[250];    
  //  vector<string> params;
  
  if(verbose_) printf("Creating RecHit tests...\n"); 


  
  return;
}

void HcalRecHitClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/RecHitMonitor/RecHit Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[150];    
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    
    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Energies",type.c_str(),type.c_str());      
    energy_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Total Energy",type.c_str(),type.c_str());      
    energyT_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
    time_[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Geo Occupancy Map",type.c_str(),type.c_str());
    occ_[i] = (TH2F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/RecHit Depth %d Occupancy Map",i);
    tot_occ_[i] = (TH2F*)infile->Get(name);
  
    
  }

  sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/RecHit Total Energy");   
  tot_energy_ = (TH1F*)infile->Get(name);

  return;
}
