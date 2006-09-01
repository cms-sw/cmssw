#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>

HcalRecHitClient::HcalRecHitClient(const ParameterSet& ps, MonitorUserInterface* mui){

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;
  for(int i=0; i<3; i++){
    occ[i]=0;
    energy[i]=0;
    energyT[i]=0;
    time[i]=0;
  }
  tot_occ=0;
  tot_energy=0;

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

}

HcalRecHitClient::HcalRecHitClient(){

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = 0;
  for(int i=0; i<3; i++){
    occ[i]=0;
    energy[i]=0;
    energyT[i]=0;
    time[i]=0;
  }
  tot_occ=0;
  tot_energy=0;

  // verbosity switch
  verbose_ = false;

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
  this->resetME();
  return;
}

void HcalRecHitClient::beginRun(void){

  if ( verbose_ ) cout << "HcalRecHitClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalRecHitClient::endJob(void) {

  if ( verbose_ ) cout << "HcalRecHitClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup(); 
  return;
}

void HcalRecHitClient::endRun(void) {

  if ( verbose_ ) cout << "HcalRecHitClient: endRun, jevt = " << jevt_ << endl;

  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalRecHitClient::setup(void) {
  return;
}

void HcalRecHitClient::cleanup(void) {

  if ( cloneME_ ) {
    for(int i=0; i<3; i++){
      if(occ[i]) delete occ[i];
      if(energy[i]) delete energy[i];
      if(energyT[i]) delete energyT[i];
      if(time[i]) delete time[i];
    }    
    if(tot_occ) delete tot_occ;
    if(tot_energy) delete tot_energy;
  }
  
  for(int i=0; i<3; i++){
    occ[i]=0; energy[i]=0;
    energyT[i]=0; time[i]=0;
  }
  tot_occ=0; tot_energy=0;
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  
  return;
}

void HcalRecHitClient::subscribe(void){

  if ( verbose_ ) cout << "HcalRecHitClient: subscribe" << endl;
  if(mui_){
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HBHE/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HF/*");
    mui_->subscribe("*/HcalMonitor/RecHitMonitor/HO/*");
  }
  return;
}

void HcalRecHitClient::subscribeNew(void){
  if(mui_){
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HBHE/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HF/*");
    mui_->subscribeNew("*/HcalMonitor/RecHitMonitor/HO/*");
  }
  return;
}

void HcalRecHitClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalRecHitClient: unsubscribe" << endl;
  if(mui_){
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HBHE/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HF/*");
    mui_->unsubscribe("*/HcalMonitor/RecHitMonitor/HO/*");
  }
  return;
}

void HcalRecHitClient::errorOutput(){
  
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

  if ( verbose_ ) cout << "HcalRecHitClient: report" << endl;
  this->setup();

  char name[256];
  sprintf(name, "Collector/%s/HcalMonitor/RecHitMonitor/RecHit Event Number",process_.c_str());
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
  int updates = mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalRecHitClient: " << updates << " updates" << endl;
  }

  return;
}

void HcalRecHitClient::getHistograms(){

  char name[150];    
  for(int i=0; i<3; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF"; 
    
    sprintf(name,"RecHitMonitor/%s/%s RecHit Energies",type.c_str(),type.c_str());      
    energy[i] = getHisto(name, process_,mui_,verbose_,cloneME_);
    
    sprintf(name,"RecHitMonitor/%s/%s RecHit Total Energy",type.c_str(),type.c_str());      
    energyT[i] = getHisto(name, process_,mui_,verbose_,cloneME_);

    sprintf(name,"RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
    time[i] = getHisto(name, process_,mui_,verbose_,cloneME_);

    sprintf(name,"RecHitMonitor/%s/%s RecHit Geo Occupancy Map",type.c_str(),type.c_str());
    occ[i] = getHisto2(name, process_,mui_,verbose_,cloneME_);
    
  }

  sprintf(name,"RecHitMonitor/RecHit Geo Occupancy Map");
  tot_occ = getHisto2(name, process_, mui_, verbose_,cloneME_);
  
  sprintf(name,"RecHitMonitor/RecHit Total Energy");   
  tot_energy = getHisto(name, process_,mui_, verbose_,cloneME_);

  return;
}

void HcalRecHitClient::resetME(){
  
  Char_t name[150];    
  MonitorElement* me;
  
  for(int i=0; i<3; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF"; 

    sprintf(name,"Collector/%s/HcalMonitor/RecHitMonitor/%s/%s RecHit Geo Occupancy Map",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    
    sprintf(name,"Collector/%s/HcalMonitor/RecHitMonitor/%s/%s RecHit Energies",process_.c_str(),type.c_str(),type.c_str());      
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    sprintf(name,"Collector/%s/HcalMonitor/RecHitMonitor/%s/%s RecHit Total Energy",process_.c_str(),type.c_str(),type.c_str());      
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    sprintf(name,"Collector/%s/HcalMonitor/RecHitMonitor/%s/%s RecHit Times",process_.c_str(),type.c_str(),type.c_str());      
    me = mui_->get(name);
    if(me) mui_->softReset(me);
  }

  sprintf(name,"Collector/%s/HcalMonitor/RecHitMonitor/RecHit Geo Occupancy Map",process_.c_str());
  me = mui_->get(name);
  if(me) mui_->softReset(me);

  sprintf(name,"Collector/%s/HcalMonitor/RecHitMonitor/RecHit Total Energy",process_.c_str());  
  me = mui_->get(name);
  if(me) mui_->softReset(me);

  return;
}


void HcalRecHitClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing HcalRecHitClient html output ..." << endl;
  string client = "RecHitMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);
  
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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal RecHits</span></h2> " << endl;
  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;
  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"RecHitMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"RecHitMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"RecHitMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal RecHit Histograms</strong></h2>" << endl;
  
  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>Global Histograms</h3></td></tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(tot_occ,"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML(tot_energy,"Total Energy (GeV)","Events", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;


  for(int i=0; i<3; i++){
    htmlFile << "<tr align=\"left\">" << endl;
    
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF"; 
    
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(occ[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML(energyT[i],"Total Energy (GeV)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(energy[i],"RecHit Energy (GeV)","Events", 92, htmlFile,htmlDir);
    histoHTML(time[i],"RecHit Time (nS)","Events", 100, htmlFile,htmlDir);
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
  char meTitle[250], name[250];    
  vector<string> params;
  
  printf("Creating RecHit tests...\n"); 
  return;

  sprintf(name, "Collector/%s/HcalMonitor/BEAM ENERGY",process_.c_str());
  MonitorElement* me = mui_->get(name);
  float beamE = -10;
  if ( me ) {
    string s = me->valueString();
    sscanf((s.substr(2,s.length()-2)).c_str(), "%f", &beamE);
  }

  if(beamE>0){    
    char nrg[100]; sprintf(nrg,"%f",beamE*beamE_thresh_);
    char nrg2[100]; sprintf(nrg,"%f",beamE*1.10);
    char err[100]; sprintf(err,"%f",beamE*beamE_width_);

    sprintf(meTitle,"Collector/%s/HcalMonitor/RecHitMonitor/RecHit Total Energy",process_.c_str()); 
    sprintf(name,"HCAL RecHit Energy to Beam Energy");  
    if( dqmQtests_.find(name) == dqmQtests_.end()){ 
      string test = ((string)name);
      MonitorElement* me = mui_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("0.367"); params.push_back("0.135");  //warn, err probs
	params.push_back(nrg);  params.push_back(err);  //mean, sigma
	params.push_back("useRMS");  // useSigma or useRMS
	createMeanValueTest(mui_, params);
      }	   
    }

    for(int i=0; i<3; i++){
      string type = "HBHE";
      if(i==1) type = "HO"; 
      if(i==2) type = "HF";
      sprintf(meTitle,"Collector/%s/HcalMonitor/RecHitMonitor/%s/%s RecHit Energies",process_.c_str(),type.c_str(),type.c_str());      
      sprintf(name,"HCAL %s Max Energy",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end()){ 
	string test = ((string)name);
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("0.367"); params.push_back("0.135");  //warn, err probs
	  params.push_back("0"); params.push_back(nrg2);  //xmin, xmax
	  createXRangeTest(mui_, params);
	}	   
      }
    }
  }
  
  return;
}

void HcalRecHitClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/RecHitMonitor/RecHit Event Number");
  string s =tnd->GetTitle();
  ievt_ = -1;
  sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);

  char name[150];    
  for(int i=0; i<3; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF"; 
    
    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Energies",type.c_str(),type.c_str());      
    energy[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Total Energy",type.c_str(),type.c_str());      
    energyT[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Times",type.c_str(),type.c_str());      
    time[i] = (TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/%s/%s RecHit Geo Occupancy Map",type.c_str(),type.c_str());
    occ[i] = (TH2F*)infile->Get(name);
    
  }

  sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/RecHit Geo Occupancy Map");
  tot_occ = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/HcalMonitor/RecHitMonitor/RecHit Total Energy");   
  tot_energy = (TH1F*)infile->Get(name);

  return;
}
