#include <DQM/HcalMonitorClient/interface/HcalLEDClient.h>

HcalLEDClient::HcalLEDClient(const ParameterSet& ps, MonitorUserInterface* mui){
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = mui;

  for(int i=0; i<4; i++){    
    rms_shape_[i]=0;
    mean_shape_[i]=0;
    rms_time_[i]=0;
    mean_time_[i]=0;
    rms_energy_[i]=0;
    mean_energy_[i]=0;

    rms_shapeD_[i]=0;
    mean_shapeD_[i]=0;
    rms_timeD_[i]=0;
    mean_timeD_[i]=0;
    rms_energyD_[i]=0;
    mean_energyD_[i]=0;

    err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  rms_thresh_ = ps.getUntrackedParameter<double>("LEDRMS_ErrThresh", 0.8);
  cout << "LED RMS error threshold set to " << rms_thresh_ << endl;
  
  mean_thresh_ = ps.getUntrackedParameter<double>("LEDMEAN_ErrThresh", 2.25);
  cout << "LED MEAN error threshold set to " << mean_thresh_ << endl;

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

HcalLEDClient::HcalLEDClient(){
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  mui_ = 0;
 for(int i=0; i<4; i++){    
    rms_shape_[i]=0;
    mean_shape_[i]=0;
    rms_time_[i]=0;
    mean_time_[i]=0;
    rms_energy_[i]=0;
    mean_energy_[i]=0;

    rms_shapeD_[i]=0;
    mean_shapeD_[i]=0;
    rms_timeD_[i]=0;
    mean_timeD_[i]=0;
    rms_energyD_[i]=0;
    mean_energyD_[i]=0;

    err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }

  // verbosity switch
  verbose_ = false;
  for(int i=0; i<4; i++) subDetsOn_[i] = false;

}

HcalLEDClient::~HcalLEDClient(){
  this->cleanup();
}

void HcalLEDClient::beginJob(const EventSetup& eventSetup){
  
  if ( verbose_ ) cout << "HcalLEDClient: beginJob" << endl;
  //  eventSetup.get<HcalDbRecord>().get(conditions_);
  
   // get the hcal mapping
  edm::ESHandle<HcalDbService> pSetup;
  eventSetup.get<HcalDbRecord>().get( pSetup );
  readoutMap_=pSetup->getHcalMapping();

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalLEDClient::beginRun(void){

  if ( verbose_ ) cout << "HcalLEDClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalLEDClient::endJob(void) {

  if ( verbose_ ) cout << "HcalLEDClient: endJob, ievt = " << ievt_ << endl;

  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalLEDClient::endRun(void) {

  if ( verbose_ ) cout << "HcalLEDClient: endRun, jevt = " << jevt_ << endl;

  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalLEDClient::setup(void) {
  return;
}

void HcalLEDClient::cleanup(void) {
  if( cloneME_ ){
    for(int i=0; i<4; i++){
      if(rms_shape_[i]) delete rms_shape_[i];
      if(mean_shape_[i]) delete mean_shape_[i];
      if(rms_time_[i]) delete rms_time_[i];
      if(mean_time_[i]) delete mean_time_[i];
      if(rms_energy_[i]) delete rms_energy_[i];
      if(mean_energy_[i]) delete mean_energy_[i];

      if(rms_shapeD_[i]) delete rms_shapeD_[i];
      if(mean_shapeD_[i]) delete mean_shapeD_[i];
      if(rms_timeD_[i]) delete rms_timeD_[i];
      if(mean_timeD_[i]) delete mean_timeD_[i];
      if(rms_energyD_[i]) delete rms_energyD_[i];
      if(mean_energyD_[i]) delete mean_energyD_[i];

      if(err_map_geo_[i]) delete err_map_geo_[i];
      if(err_map_elec_[i]) delete err_map_elec_[i];
      if(avg_shape_[i]) delete avg_shape_[i];
      if(avg_time_[i]) delete avg_time_[i];
      if(avg_energy_[i]) delete avg_energy_[i];
    }
  }

  for(int i=0; i<4; i++){    
    rms_shape_[i]=0;
    mean_shape_[i]=0;
    rms_time_[i]=0;
    mean_time_[i]=0;
    rms_energy_[i]=0;
    mean_energy_[i]=0;

    rms_shapeD_[i]=0;
    mean_shapeD_[i]=0;
    rms_timeD_[i]=0;
    mean_timeD_[i]=0;
    rms_energyD_[i]=0;
    mean_energyD_[i]=0;

    err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }


  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  return;
}

void HcalLEDClient::subscribe(void){

  if ( verbose_ ) cout << "HcalLEDClient: subscribe" << endl;
  if(mui_) mui_->subscribe("*/HcalMonitor/LEDMonitor/*");
  return;
}

void HcalLEDClient::subscribeNew(void){
  if(mui_) mui_->subscribeNew("*/HcalMonitor/LEDMonitor/*");
  return;
}

void HcalLEDClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalLEDClient: unsubscribe" << endl;
  if(mui_) mui_->unsubscribe("*/HcalMonitor/LEDMonitor/*");
  return;
}

void HcalLEDClient::errorOutput(){
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
	vector<QReport*> report = me->getQOthers();
	dqmReportMapOther_[meName] = report;
      }
    }
  }
  printf("LED Task: %d errs, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());

  return;
}

void HcalLEDClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalLEDClient::report(){
   if(!mui_) return;
  if ( verbose_ ) cout << "HcalLEDClient: report" << endl;
  this->setup();

  char name[256];
  sprintf(name, "%sHcalMonitor/LEDMonitor/LED Task Event Number",process_.c_str());
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

void HcalLEDClient::getHistograms(){
  if(!mui_) return;
  char name[256];

  MonitorElement* meDepTimeMean[4];
  MonitorElement* meDepTimeRMS[4];
  MonitorElement* meDepShapeMean[4];
  MonitorElement* meDepShapeRMS[4];
  MonitorElement* meDepEnergyMean[4];
  MonitorElement* meDepEnergyRMS[4];

  for(int i=0; i<4; i++){
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED Mean Time Depth %d",process_.c_str(),i+1);
    meDepTimeMean[i] = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED RMS Time Depth %d",process_.c_str(),i+1);
    meDepTimeRMS[i] = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED Mean Shape Depth %d",process_.c_str(),i+1);
    meDepShapeMean[i] = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED RMS Shape Depth %d",process_.c_str(),i+1);
    meDepShapeRMS[i] = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED Mean Energy Depth %d",process_.c_str(),i+1);
    meDepEnergyMean[i] = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED RMS Energy Depth %d",process_.c_str(),i+1);
    meDepEnergyRMS[i] = mui_->get(name);

    if(meDepTimeMean[i]) mui_->softReset(meDepTimeMean[i]);
    if(meDepTimeRMS[i]) mui_->softReset(meDepTimeRMS[i]);
    if(meDepEnergyMean[i]) mui_->softReset(meDepEnergyMean[i]);
    if(meDepEnergyRMS[i]) mui_->softReset(meDepEnergyRMS[i]);
    if(meDepShapeMean[i]) mui_->softReset(meDepShapeMean[i]);
    if(meDepShapeRMS[i]) mui_->softReset(meDepShapeRMS[i]);
  }

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF";
    if(i==3) type = "HO";

    sprintf(name,"LEDMonitor/%s/%s Average Pulse Shape",type.c_str(),type.c_str());      
    avg_shape_[i] = getHisto(name, process_,mui_,verbose_,cloneME_);
    sprintf(name,"LEDMonitor/%s/%s Average Pulse Time",type.c_str(),type.c_str());      
    avg_time_[i] = getHisto(name, process_,mui_,verbose_,cloneME_);
    sprintf(name,"LEDMonitor/%s/%s Average Pulse Energy",type.c_str(),type.c_str());      
    avg_energy_[i] = getHisto(name, process_,mui_,verbose_,cloneME_);
    
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Shape RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meShapeRMS  = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Shape Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meShapeMean  = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Time RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meTimeRMS  = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Time Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meTimeMean  = mui_->get(name);

    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Energy RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meEnergyRMS  = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Energy Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meEnergyMean  = mui_->get(name);

    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meGeoErr  = mui_->get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Elec Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meElecErr  = mui_->get(name);


    if(!meShapeRMS || !meShapeMean) return;
    if(!meTimeRMS || !meTimeMean) return;
    if(!meEnergyRMS || !meEnergyMean) return;
    mui_->softReset(meShapeRMS); mui_->softReset(meShapeMean);
    mui_->softReset(meTimeRMS); mui_->softReset(meTimeMean);
    mui_->softReset(meEnergyRMS); mui_->softReset(meEnergyMean);
    mui_->softReset(meGeoErr); mui_->softReset(meElecErr);
    
    for(int ieta=-42; ieta<=42; ieta++){
      if(ieta==0) continue;
      for(int iphi=1; iphi<=73; iphi++){
	for(int depth=1; depth<=4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",
		  process_.c_str(), type.c_str(),type.c_str(),ieta,iphi,depth);  
	  MonitorElement* me = mui_->get(name);
	  if(me){
	    meShapeRMS->Fill(me->getRMS());
	    meShapeMean->Fill(me->getMean());
	    meDepShapeRMS[depth-1]->Fill(ieta,iphi,me->getRMS());
	    meDepShapeMean[depth-1]->Fill(ieta,iphi,me->getMean());

	    if(me->getRMS()<rms_thresh_ || me->getMean()>mean_thresh_){
	      HcalSubdetector subdet = HcalBarrel;
	      if(i==1) subdet = HcalOuter;
	      else if(i==2) subdet = HcalForward;
	      else if(i==3) subdet = HcalEndcap;
	      
	      HcalDetId id(subdet,ieta,iphi,depth);
	      HcalElectronicsId eid = readoutMap_->lookup(id);
	      meGeoErr->Fill(ieta,iphi);
	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());	      
	    }
	  }
	  
	  sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",process_.c_str(),
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = mui_->get(name);
	  if(me){
	    meTimeRMS->Fill(me->getRMS());
	    meTimeMean->Fill(me->getMean());
	    meDepTimeRMS[depth-1]->Fill(ieta,iphi,me->getRMS());
	    meDepTimeMean[depth-1]->Fill(ieta,iphi,me->getMean());
	  }	  

	  sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",process_.c_str(),
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = mui_->get(name);
	  if(me){
	    meEnergyRMS->Fill(me->getRMS());
	    meEnergyMean->Fill(me->getMean());
	    meDepEnergyRMS[depth-1]->Fill(ieta,iphi,me->getRMS());
	    meDepEnergyMean[depth-1]->Fill(ieta,iphi,me->getMean());
	  }	  
	}
      }
    }
    
    rms_shape_[i]=getHisto(meShapeRMS,verbose_,cloneME_);
    mean_shape_[i]=getHisto(meShapeMean,verbose_,cloneME_);
    rms_time_[i]=getHisto(meTimeRMS,verbose_,cloneME_);
    mean_time_[i]=getHisto(meTimeMean,verbose_,cloneME_);
    rms_energy_[i]=getHisto(meEnergyRMS,verbose_,cloneME_);
    mean_energy_[i]=getHisto(meEnergyMean,verbose_,cloneME_);

    
    err_map_geo_[i]=getHisto2(meGeoErr,verbose_,cloneME_);
    err_map_elec_[i]=getHisto2(meElecErr,verbose_,cloneME_);
  }
  for(int i=0; i<4; i++){
    rms_shapeD_[i]=getHisto2(meDepShapeRMS[i],verbose_,cloneME_);
    mean_shapeD_[i]=getHisto2(meDepShapeMean[i],verbose_,cloneME_);
    rms_timeD_[i]=getHisto2(meDepTimeRMS[i],verbose_,cloneME_);
    mean_timeD_[i]=getHisto2(meDepTimeMean[i],verbose_,cloneME_);
    rms_energyD_[i]=getHisto2(meDepEnergyRMS[i],verbose_,cloneME_);
    mean_energyD_[i]=getHisto2(meDepEnergyMean[i],verbose_,cloneME_);
  }

  return;
}

void HcalLEDClient::analyze(void){
  
  jevt_++;
  int updates = 0;
  if(mui_) mui_->getNumUpdates();
  if ( (updates % 10) == 0 ) {
    if ( verbose_ ) cout << "HcalLEDClient: " << updates << " updates" << endl;
  }
  
  return;
}

void HcalLEDClient::createTests(){
  if(!mui_) return;
  
  char meTitle[250], name[250];    
  vector<string> params;

  if(verbose_) printf("Creating LED tests...\n");
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF";
    if(i==3) type = "HO";

    sprintf(meTitle,"%sHcalMonitor/LEDMonitor/%s/%s LED Shape RMS Values",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s LED Shape RMS Values: X-Range",type.c_str());
    if( dqmQtests_.find(name) == dqmQtests_.end() ){	
      MonitorElement* me = mui_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	char low[20];
	sprintf(low,"%.2f\n",rms_thresh_);
	params.push_back(low); params.push_back("2");  //xmin, xmax
	createXRangeTest(mui_, params);
      }
    }
    
    sprintf(meTitle,"%sHcalMonitor/LEDMonitor/%s/%s LED Shape Mean Values",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s LED Shape RMS Values: X-Range",type.c_str());
    if( dqmQtests_.find(name) == dqmQtests_.end() ){	
      MonitorElement* me = mui_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	char high[20];
	sprintf(high,"%.2f\n",mean_thresh_);
	params.push_back("0"); params.push_back(high);  //xmin, xmax
	createXRangeTest(mui_, params);
      }
    }    
  }

  return;
}

void HcalLEDClient::resetME(){
  if(!mui_) return;
  Char_t name[150];    
  MonitorElement* me;
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO"; 
    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s Average Pulse Shape",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);

    sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s Average Pulse Time",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);
    
    for(int ieta=-42; ieta<42; ieta++){
      for(int iphi=0; iphi<72; iphi++){
	for(int depth=1; depth<4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",
		  process_.c_str(), type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = mui_->get(name);
	  if(me) mui_->softReset(me);

	  sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",process_.c_str(),
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = mui_->get(name);
	  if(me) mui_->softReset(me);

	  sprintf(name,"%sHcalMonitor/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",process_.c_str(),
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = mui_->get(name);
	  if(me) mui_->softReset(me);
	}
      }
    }
  }
  return;
}

void HcalLEDClient::htmlOutput(int run, string htmlDir, string htmlName){
  
  cout << "Preparing HcalLEDClient html output ..." << endl;
  string client = "LEDMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal LED Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal LEDs</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<table  width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"LEDMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"LEDMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"LEDMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Hcal LED Histograms</strong></h2>" << endl;
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
  histoHTML2(mean_timeD_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_timeD_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_timeD_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_timeD_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_timeD_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_timeD_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_timeD_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_timeD_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_energyD_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_energyD_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_energyD_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_energyD_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_energyD_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_energyD_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_energyD_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_energyD_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_shapeD_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_shapeD_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_shapeD_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_shapeD_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_shapeD_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_shapeD_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(mean_shapeD_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(rms_shapeD_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
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
     histoHTML2(err_map_geo_[i],"iEta","iPhi", 92, htmlFile,htmlDir);
     histoHTML2(err_map_elec_[i],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;
     
     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(avg_shape_[i],"Average Pulse Shape","Events", 92, htmlFile,htmlDir);
     histoHTML(avg_time_[i],"Average Pulse Time","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;
     
     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(avg_energy_[i],"Average ADC Sum","Events", 92, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(rms_shape_[i],"Shape RMS Value","Events", 92, htmlFile,htmlDir);
     histoHTML(mean_shape_[i],"Shape Mean Value","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(rms_time_[i],"Time RMS Value","Events", 92, htmlFile,htmlDir);
     histoHTML(mean_time_[i],"Time Mean Value","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(rms_energy_[i],"ADC Sum RMS Value","Events", 92, htmlFile,htmlDir);
     histoHTML(mean_energy_[i],"ADC Sum Mean Value","Events", 100, htmlFile,htmlDir);
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


void HcalLEDClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/LEDMonitor/LED Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  char name[256];
  for(int i=0; i<4; i++){
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED Mean Time Depth %d",process_.c_str(),i+1);
    mean_timeD_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED RMS Time Depth %d",process_.c_str(),i+1);
    rms_timeD_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED Mean Shape Depth %d",process_.c_str(),i+1);
    mean_shapeD_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED RMS Shape Depth %d",process_.c_str(),i+1);
    rms_shapeD_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED Mean Energy Depth %d",process_.c_str(),i+1);
    mean_energyD_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcalMonitor/LEDMonitor/LED RMS Energy Depth %d",process_.c_str(),i+1);
    rms_energyD_[i]=(TH2F*)infile->Get(name);
  }


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue; 
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF"; 
    if(i==3) type = "HO";


    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s Average Pulse Shape",type.c_str(),type.c_str());      
    avg_shape_[i] = (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s Average Pulse Time",type.c_str(),type.c_str());      
    avg_time_[i] = (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s Average Pulse Energy",type.c_str(),type.c_str());      
    avg_energy_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Shape RMS Values",type.c_str(),type.c_str());
    rms_shape_[i]=(TH1F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Shape Mean Values",type.c_str(),type.c_str());
    mean_shape_[i]=(TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Time RMS Values",type.c_str(),type.c_str());
    rms_time_[i]=(TH1F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Time Mean Values",type.c_str(),type.c_str());
    mean_time_[i]=(TH1F*)infile->Get(name);

    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Energy RMS Values",type.c_str(),type.c_str());
    rms_energy_[i]=(TH1F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Energy Mean Values",type.c_str(),type.c_str());
    mean_energy_[i]=(TH1F*)infile->Get(name);


    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Geo Error Map",type.c_str(),type.c_str());
    err_map_geo_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Elec Error Map",type.c_str(),type.c_str());
    err_map_elec_[i]=(TH2F*)infile->Get(name);


    for(int ieta=-42; ieta<=42; ieta++){
      if(ieta==0) continue;
      for(int iphi=1; iphi<=73; iphi++){
	for(int depth=1; depth<=4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  TH1F* h = (TH1F*)infile->Get(name);
	  if(h){
	    rms_shape_[i]->Fill(h->GetRMS());
	    mean_shape_[i]->Fill(h->GetMean());
	  }
	  
	  sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  h = (TH1F*)infile->Get(name);
	  if(h){
	    rms_time_[i]->Fill(h->GetRMS());
	    mean_time_[i]->Fill(h->GetMean());
	  }	  

	  sprintf(name,"DQMData/HcalMonitor/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  h = (TH1F*)infile->Get(name);
	  if(h){
	    rms_energy_[i]->Fill(h->GetRMS());
	    mean_energy_[i]->Fill(h->GetMean());
	  }	  
	}
      }
    }


  }

  return;
}
