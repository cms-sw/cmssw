#include <DQM/HcalMonitorClient/interface/HcalPedestalClient.h>

HcalPedestalClient::HcalPedestalClient(const ParameterSet& ps, MonitorUserInterface* mui){
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  ///HB ieta/iphi/depths
  etaMin[0]=1; etaMax[0]=16;
  phiMin[0]=1; phiMax[0]=71;
  depMin[0]=1; depMax[0]=2;
  
  ///HO ieta/iphi/depths
  etaMin[1]=1; etaMax[1]=15;
  phiMin[1]=1; phiMax[1]=71;
  depMin[1]=4; depMax[1]=4;

  ///HF ieta/iphi/depths
  etaMin[2]=29; etaMax[2]=41;
  phiMin[2]=1; phiMax[2]=71;
  depMin[2]=1; depMax[2]=1;

  ///HE ieta/iphi/depths
  etaMin[3]=16; etaMax[3]=29;
  phiMin[3]=1; phiMax[3]=71;
  depMin[3]=1; depMax[3]=3;

  mui_ = mui;
  readoutMap_=0;

  for(int i=0; i<3; i++){
    all_peds[i]=0;   ped_rms[i]=0;
    ped_mean[i]=0;   capid_rms[i]=0;
    capid_mean[i]=0; qie_rms[i]=0;
    qie_mean[i]=0;   err_map_geo[i]=0;
    err_map_elec[i]=0;
  }
  
  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  // per channel tests switch
  doPerChanTests_ = ps.getUntrackedParameter<bool>("DoPerChanTests", false);
  // DQM default process name
  process_ = ps.getUntrackedParameter<string>("processName", "HcalMonitor");

  pedrms_thresh_ = ps.getUntrackedParameter<double>("PedestalRMS_ErrThresh", 1);
  cout << "Pedestal RMS error threshold set to " << pedrms_thresh_ << endl;

  pedmean_thresh_ = ps.getUntrackedParameter<double>("PedestalMEAN_ErrThresh", 2);
  cout << "Pedestal MEAN error threshold set to " << pedmean_thresh_ << endl;

  caprms_thresh_ = ps.getUntrackedParameter<double>("CapIdRMS_ErrThresh", 0.25);
  cout << "CapId RMS Variance error threshold set to " << caprms_thresh_ << endl;

  capmean_thresh_ = ps.getUntrackedParameter<double>("CapIdMEAN_ErrThresh", 1.5);
  cout << "CapId MEAN Variance error threshold set to " << capmean_thresh_ << endl;

}

HcalPedestalClient::HcalPedestalClient(){
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  ///HB ieta/iphi/depths
  etaMin[0]=1; etaMax[0]=16;
  phiMin[0]=1; phiMax[0]=71;
  depMin[0]=1; depMax[0]=2;
  
  ///HO ieta/iphi/depths
  etaMin[1]=1; etaMax[1]=15;
  phiMin[1]=1; phiMax[1]=71;
  depMin[1]=4; depMax[1]=4;

  ///HF ieta/iphi/depths
  etaMin[2]=29; etaMax[2]=41;
  phiMin[2]=1; phiMax[2]=71;
  depMin[2]=1; depMax[2]=1;

  ///HE ieta/iphi/depths
  etaMin[3]=16; etaMax[3]=29;
  phiMin[3]=1; phiMax[3]=71;
  depMin[3]=1; depMax[3]=3;

  mui_ = 0;
  readoutMap_=0;

  for(int i=0; i<3; i++){
    all_peds[i]=0;   ped_rms[i]=0;
    ped_mean[i]=0;   capid_rms[i]=0;
    capid_mean[i]=0; qie_rms[i]=0;
    qie_mean[i]=0;   err_map_geo[i]=0;
    err_map_elec[i]=0;
  }
  
  // verbosity switch
  verbose_ = false;
  offline_ = true;

}

HcalPedestalClient::~HcalPedestalClient(){

  this->cleanup();

}

void HcalPedestalClient::beginJob(const EventSetup& eventSetup){

  if ( verbose_ ) cout << "HcalPedestalClient: beginJob" << endl;
  eventSetup.get<HcalDbRecord>().get(conditions_);

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

void HcalPedestalClient::beginRun(void){

  if ( verbose_ ) cout << "HcalPedestalClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->subscribe();
  this->resetME();
  return;
}

void HcalPedestalClient::endJob(void) {

  if ( verbose_ ) cout << "HcalPedestalClient: endJob, ievt = " << ievt_ << endl;

  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalPedestalClient::endRun(void) {

  if ( verbose_ ) cout << "HcalPedestalClient: endRun, jevt = " << jevt_ << endl;

  //  this->resetME();
  //  this->unsubscribe();
  this->cleanup();
  return;
}

void HcalPedestalClient::setup(void) {

  return;
}

void HcalPedestalClient::cleanup(void) {
  if(cloneME_){
    for(int i=0; i<3; i++){
      if(all_peds[i]); delete all_peds[i];   
      if(ped_rms[i]); delete ped_rms[i];
      if(ped_mean[i]); delete ped_mean[i];   
      if(capid_rms[i]); delete capid_rms[i];
      if(capid_mean[i]); delete capid_mean[i]; 
      if(qie_rms[i]); delete qie_rms[i];
      if(qie_mean[i]); delete qie_mean[i];   
      if(err_map_geo[i]); delete err_map_geo[i];
      if(err_map_elec[i]); delete err_map_elec[i];
    }
  }
  for(int i=0; i<3; i++){
    all_peds[i]=0;   ped_rms[i]=0;
    ped_mean[i]=0;   capid_rms[i]=0;
    capid_mean[i]=0; qie_rms[i]=0;
    qie_mean[i]=0;   err_map_geo[i]=0;
    err_map_elec[i]=0;
  }
  
  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalPedestalClient::subscribe(void){

  if ( verbose_ ) cout << "HcalPedestalClient: subscribe" << endl;
  if(mui_) mui_->subscribe("*/HcalMonitor/PedestalMonitor/*");
  return;
}

void HcalPedestalClient::subscribeNew(void){
  if(mui_) mui_->subscribeNew("*/HcalMonitor/PedestalMonitor/*");
  return;
}

void HcalPedestalClient::unsubscribe(void){

  if ( verbose_ ) cout << "HcalPedestalClient: unsubscribe" << endl;
  if(mui_) mui_->unsubscribe("*/HcalMonitor/PedestalMonitor/*");
  return;
}

void HcalPedestalClient::errorOutput(){

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
  printf("Pedestal Task: %d errs, %d warnings, %d others\n",dqmReportMapErr_.size(),dqmReportMapWarn_.size(),dqmReportMapOther_.size());
  
  return;
}

void HcalPedestalClient::getErrors(map<string, vector<QReport*> > outE, map<string, vector<QReport*> > outW, map<string, vector<QReport*> > outO){

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

void HcalPedestalClient::report(){

  if ( verbose_ ) cout << "HcalPedestalClient: report" << endl;
  this->setup();
  
    char name[256];    
  sprintf(name, "Collector/%s/HcalMonitor/PedestalMonitor/Pedestal Task Event Number",process_.c_str());
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
  
void HcalPedestalClient::getHistograms(){
  char name[256];    
  for(int i=0; i<4; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF";
    if(i==3) type = "HBHE";
    
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* mePedRMS  = mui_->get(name);
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* mePedMean = mui_->get(name);
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s CapID RMS Variance",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meCapRMS  = mui_->get(name);
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s CapID Mean Variance",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meCapMean = mui_->get(name);
    
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s QIE RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meQieRMS  = mui_->get(name);
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s QIE Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meQieMean = mui_->get(name);
    
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meGeoErr  = mui_->get(name);
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Elec Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meElecErr = mui_->get(name);
    
    if(!mePedRMS || !mePedMean) return;
    if(!meCapRMS || !meCapMean) return;
    if(!meQieRMS || !meQieMean) return;
    if(i<3 && mui_){
      mui_->softReset(mePedRMS); mui_->softReset(mePedMean);
      mui_->softReset(meCapRMS); mui_->softReset(meCapMean);
      mui_->softReset(meQieRMS); mui_->softReset(meQieMean);
      mui_->softReset(meGeoErr); mui_->softReset(meElecErr);
    }
    bool capidOK = false;
    for(int ieta=-etaMax[i]; ieta<=etaMax[i]; ieta++){
      if(abs(ieta)<etaMin[i]) continue;
      
      for(int iphi=phiMin[i]; iphi<=phiMax[i]; iphi++){
	for(int depth=depMin[i]; depth<=depMax[i]; depth++){
	  if(i==0 && abs(ieta)==16 && depth==3) continue;
	  if(i==3 && abs(ieta)==16 && (depth==1 || depth==2)) continue;
	  capidOK = true;
	  
	  HcalSubdetector subdet = HcalBarrel;
	  if(i==1) subdet = HcalOuter;
	  else if(i==2) subdet = HcalForward;
	  else if(i==3) subdet = HcalEndcap;
	  HcalDetId id(subdet,ieta,iphi,depth);
	  HcalElectronicsId eid = readoutMap_->lookup(id);
	  	  
	  float capmean[4]; float caprms[4];	  
	  for(int capid=0; capid<4; capid++){
	    capmean[capid]=0; caprms[capid]=0; 
	    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Value ieta=%d iphi=%d depth=%d CAPID=%d",process_.c_str(),
		    type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    MonitorElement* me = mui_->get(name);
	    if(me!=NULL){
	      capmean[capid] = me->getMean();
	      caprms[capid] = me->getRMS();
	      mePedRMS->Fill(me->getRMS());
	      mePedMean->Fill(me->getMean());
	      //	      printf("%f, %f\n",capmean[capid],caprms[capid]);
	      if(me->getRMS()>pedrms_thresh_ || me->getMean()<pedmean_thresh_) {
		int idx=i;
		if(i==3) idx=0;
		meGeoErr->Fill(ieta,iphi);
		meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	      }
	    }
	    else capidOK=false;
	  }
	  if(capidOK){
	    meCapMean->Fill(maxDiff(capmean[0],capmean[1],capmean[2],capmean[3]));
	    meCapRMS->Fill(maxDiff(caprms[0],caprms[1],caprms[2],caprms[3]));
	    
	    if(maxDiff(capmean[0],capmean[1],capmean[2],capmean[3])>capmean_thresh_){
	      meGeoErr->Fill(ieta,iphi);
	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	    if(maxDiff(caprms[0],caprms[1],caprms[2],caprms[3])>caprms_thresh_){
	      meGeoErr->Fill(ieta,iphi);
	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	       
	    float avg = (capmean[0]+capmean[1]+capmean[2]+capmean[3])/4.0;
	    meQieMean->Fill(avg);
	    avg = (caprms[0]+caprms[1]+caprms[2]+caprms[3])/4.0;
	    meQieRMS->Fill(avg);
	  }
	}
      }      
    }

    int idx = i;
    if(i>2) idx=0;
    
    sprintf(name,"PedestalMonitor/%s/%s All Pedestal Values",type.c_str(),type.c_str());      
    all_peds[idx] = getHisto(name, process_,mui_,verbose_,cloneME_);
    
    ped_rms[idx] = getHisto(mePedRMS,verbose_,cloneME_);
    ped_mean[idx] = getHisto(mePedMean,verbose_,cloneME_);
    
    capid_rms[idx] = getHisto(meCapRMS,verbose_,cloneME_);
    capid_mean[idx] = getHisto(meCapMean,verbose_,cloneME_);
    
    qie_rms[idx] = getHisto(meQieRMS,verbose_,cloneME_);
    qie_mean[idx] = getHisto(meQieMean,verbose_,cloneME_);
    
    err_map_geo[idx] = getHisto2(meGeoErr,verbose_,cloneME_);
    err_map_elec[idx] = getHisto2(meElecErr,verbose_,cloneME_);
    
  }

  return;
}

void HcalPedestalClient::analyze(void){

  jevt_++;
  int updates = mui_->getNumUpdates();
  if ( updates % 10 == 0 ) {
    if ( verbose_ ) cout << "HcalPedestalClient: " << updates << " updates" << endl;
  }
  

  return;
}

void HcalPedestalClient::createTests(){
  char meTitle[250], name[250];    
  vector<string> params;
  
  printf("Creating Pedestal tests...\n");
  
  for(int i=0; i<4; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF";
    if(i==3) type = "HBHE";
    
    if(i<3){
      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s Pedestal Errors by Geometry",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back((string)meTitle); params.push_back((string)name);  //hist and qtest titles
	  params.push_back("0"); params.push_back("1e-10");  //mean ranges
	  params.push_back("0"); params.push_back("1e-10");  //rms ranges
	  createH2ContentTest(mui_, params);
	}
      }
      
      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s All Pedestal Values",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s All Pedestal Values: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	  params.push_back("0"); params.push_back("10");  //xmin, xmax
	  createXRangeTest(mui_, params);
	}
      }

      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal RMS Values",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s Pedestal RMS Values: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	  char high[20];
	  sprintf(high,"%.2f\n",pedrms_thresh_);
	  params.push_back("0"); params.push_back(high);  //xmin, xmax
	  createXRangeTest(mui_, params);
	}
      }
      
      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Mean Values",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s Pedestal Mean Values: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	  char low[20];
	  sprintf(low,"%.2f\n",pedmean_thresh_);
	  params.push_back(low); params.push_back("10");  //xmin, xmax
	  createXRangeTest(mui_, params);
	}
      }

      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s CapID RMS Variance",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s CapId RMS Variance: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	  char high[20];
	  sprintf(high,"%.2f\n",caprms_thresh_);
	  params.push_back("0"); params.push_back(high);  //xmin, xmax
	  createXRangeTest(mui_, params);
	}
      }
      
      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s CapID Mean Variance",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s CapId Mean Variance: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = mui_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	  char high[20];
	  sprintf(high,"%.2f\n",capmean_thresh_);
	  params.push_back("0"); params.push_back(high);  //xmin, xmax
	  createXRangeTest(mui_, params);
	}
      }

    }
    
    if(doPerChanTests_){
      for(int ieta=-etaMax[i]; ieta<=etaMax[i]; ieta++){
	if(abs(ieta)<etaMin[i]) continue;
	if(ieta==0) continue;
	for(int iphi=phiMin[i]; iphi<=phiMax[i]; iphi++){
	  for(int depth=depMin[i]; depth<=depMax[i]; depth++){
	    if(i==0 && abs(ieta)==16 && depth==3) continue;
	    if(i==3 && abs(ieta)==16 && (depth==1 || depth==2)) continue;
	    
	    HcalSubdetector subdet = HcalBarrel;
	    if(i==1) subdet = HcalOuter;
	    else if(i==2) subdet = HcalForward;
	    else if(i==3) subdet = HcalEndcap;
	    const HcalDetId id(subdet,ieta,iphi,depth);
	    bool qie = true;
	    for(int capid=0; capid<4 && qie; capid++){
	      sprintf(meTitle,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Value ieta=%d iphi=%d depth=%d CAPID=%d", process_.c_str(),type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	      sprintf(name,"%s Pedestal ieta=%d iphi=%d depth=%d CAPID=%d: Sigma",type.c_str(),ieta,iphi,depth,capid);  
	      if( dqmQtests_.find(name) == dqmQtests_.end()){ 
		string test = ((string)name);
		MonitorElement* me = mui_->get(meTitle);
		if(me!=NULL){
		  dqmQtests_[name]=meTitle;			
		  float mean = 0; float width = 1;
		  const HcalPedestal* pedm = (*conditions_).getPedestal(id);
		  if(pedm) mean = pedm->getValue(capid);
		  const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
		  if(pedw) width = pedw->getWidth(capid);
		  
		  params.clear();
		  params.push_back(meTitle); params.push_back(name);  //hist and test titles
		  params.push_back("0.367"); params.push_back("0.135");  //warn, err probs
		  char m[100]; sprintf(m,"%f",mean);
		  char w[100]; sprintf(w,"%f",width);
		  params.push_back(m);  params.push_back(w);  //mean, sigma
		  params.push_back("useSigma");  // useSigma or useRMS
		  createMeanValueTest(mui_, params);
		}
		else{ qie = false; }//didn't find this qie, so the next three aren't there either...
	      }
	      else{ qie =  false; }//already have this qie, so we already have the others...
	    }
	  }
	}
      }    
    }
  }

  return;
}

void HcalPedestalClient::resetME(){
  
  Char_t name[150];    
  MonitorElement* me;

  for(int i=0; i<3; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF"; 
    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s All Pedestal Values",process_.c_str(),type.c_str(),type.c_str());
    me = mui_->get(name);
    if(me) mui_->softReset(me);

    for(int ieta=-42; ieta<42; ieta++){
      for(int iphi=0; iphi<72; iphi++){
	for(int depth=0; depth<4; depth++){
	  for(int capid=0; capid<4; capid++){
	    sprintf(name,"Collector/%s/HcalMonitor/PedestalMonitor/%s/%s Pedestal Value ieta=%d iphi=%d depth=%d CAPID=%d",process_.c_str(),
		    type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    me = mui_->get(name);
	    if(me) mui_->softReset(me);
	  }
	}
      }
    }
    if(all_peds[i]) all_peds[i]->Reset();
    if(ped_rms[i])ped_rms[i]->Reset();
    if(ped_mean[i])ped_mean[i]->Reset();
    if(capid_mean[i])capid_mean[i]->Reset();
    if(capid_rms[i])capid_rms[i]->Reset();
    if(qie_mean[i])qie_mean[i]->Reset();
    if(qie_rms[i])qie_rms[i]->Reset();
    if(err_map_geo[i])err_map_geo[i]->Reset();
    if(err_map_elec[i])err_map_elec[i]->Reset();
  }

  return;
}

void HcalPedestalClient::htmlOutput(int run, string htmlDir, string htmlName){
  
  cout << "Preparing HcalPedestalClient html output ..." << endl;
  string client = "PedestalMonitor";
  htmlErrors(htmlDir,client,process_,mui_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

  ofstream htmlFile;
  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor: Hcal Pedestal Task output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Pedestals</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<table border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"PedestalMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"PedestalMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"PedestalMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "</tr></table>" << endl;
  htmlFile << "<hr>" << endl;
  
  htmlFile << "<h2><strong>Hcal Pedestal Histograms</strong></h2>" << endl;
  htmlFile << "<h3>" << endl;
  htmlFile << "<a href=\"#HBHE_Plots\">HB-HE Plots </a></br>" << endl;
  htmlFile << "<a href=\"#HO_Plots\">HO Plots </a></br>" << endl;
  htmlFile << "<a href=\"#HF_Plots\">HF Plots </a></br>" << endl;
  htmlFile << "</h3>" << endl;
  htmlFile << "<hr>" << endl;

  htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
  htmlFile << "cellpadding=\"10\"> " << endl;
  
  for(int i=0; i<3; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF"; 

    htmlFile << "<tr align=\"left\">" << endl;    
    htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(err_map_geo[i],"iEta","iPhi", 92, htmlFile,htmlDir);
    histoHTML2(err_map_elec[i],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(ped_rms[i],"Pedestal RMS (fC)","Events", 92, htmlFile,htmlDir);
    histoHTML(ped_mean[i],"Pedestal Mean (fC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(capid_rms[i],"Variance in CAPID RMS (fC)","Events", 92, htmlFile,htmlDir);

    histoHTML(capid_mean[i],"Variance in CAPID Mean (fC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(qie_rms[i],"Average QIE RMS (fC)","Events", 92, htmlFile,htmlDir);
    histoHTML(qie_mean[i],"Average QIE Mean (fC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(all_peds[i],"Pedestal Value (fC)","Events", 92, htmlFile,htmlDir);
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



void HcalPedestalClient::loadHistograms(TFile* infile){

  TNamed* tnd = (TNamed*)infile->Get("DQMData/HcalMonitor/PedestalMonitor/Pedestal Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[256];    
  for(int i=0; i<4; i++){
    string type = "HBHE";
    if(i==1) type = "HO"; 
    if(i==2) type = "HF";
    if(i==3) type = "HBHE";

    if(i<3){
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s All Pedestal Values",type.c_str(),type.c_str());
      all_peds[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal RMS Values",type.c_str(),type.c_str());
      ped_rms[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal Mean Values",type.c_str(),type.c_str());
      ped_mean[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s CapID RMS Variance",type.c_str(),type.c_str());
      capid_rms[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s CapID Mean Variance",type.c_str(),type.c_str());
      capid_mean[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s QIE RMS Values",type.c_str(),type.c_str());
      qie_rms[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s QIE Mean Values",type.c_str(),type.c_str());
      qie_mean[i] = (TH1F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal Geo Error Map",type.c_str(),type.c_str());
      err_map_geo[i] = (TH2F*)infile->Get(name);
      
      sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal Elec Error Map",type.c_str(),type.c_str());
      err_map_elec[i] = (TH2F*)infile->Get(name);
    }
    int idx=i;
    if(i>2) idx = 2;

    bool capidOK = false;
    for(int ieta=-etaMax[i]; ieta<=etaMax[i]; ieta++){
      if(abs(ieta)<etaMin[i]) continue;
      
      for(int iphi=phiMin[i]; iphi<=phiMax[i]; iphi++){
	for(int depth=depMin[i]; depth<=depMax[i]; depth++){
	  if(i==0 && abs(ieta)==16 && depth==3) continue;
	  if(i==3 && abs(ieta)==16 && (depth==1 || depth==2)) continue;
	  capidOK = true;
	  
	  HcalSubdetector subdet = HcalBarrel;
	  if(i==1) subdet = HcalOuter;
	  else if(i==2) subdet = HcalForward;
	  else if(i==3) subdet = HcalEndcap;
	  //	  HcalDetId id(subdet,ieta,iphi,depth);
	  //	  HcalElectronicsId eid = readoutMap_->lookup(id);
	  
	  float capmean[4]; float caprms[4];	  
	  for(int capid=0; capid<4; capid++){
	    capmean[capid]=0; caprms[capid]=0; 
	    sprintf(name,"DQMData/HcalMonitor/PedestalMonitor/%s/%s Pedestal Value ieta=%d iphi=%d depth=%d CAPID=%d", type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    TH1F* me = (TH1F*)infile->Get(name);
	    
	    if(me!=NULL){
	      capmean[capid] = me->GetMean();
	      caprms[capid] = me->GetRMS();
	      if(ped_rms[idx]) ped_rms[idx]->Fill(me->GetRMS());
	      if(ped_mean[idx]) ped_mean[idx]->Fill(me->GetMean());
	      //	      printf("%f, %f\n",capmean[capid],caprms[capid]);
	      if(me->GetRMS()>pedrms_thresh_ || me->GetMean()<pedmean_thresh_) {
		if(err_map_geo[idx])err_map_geo[idx]->Fill(ieta,iphi);
		//		meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	      }
	    }
	    else capidOK=false;
	  }
	  if(capidOK){
	    if(capid_mean[idx]) capid_mean[idx]->Fill(maxDiff(capmean[0],capmean[1],capmean[2],capmean[3]));
	    if(capid_rms[idx]) capid_rms[idx]->Fill(maxDiff(caprms[0],caprms[1],caprms[2],caprms[3]));
	    
	    if(maxDiff(capmean[0],capmean[1],capmean[2],capmean[3])>capmean_thresh_){
	      if(err_map_geo[idx]) err_map_geo[idx]->Fill(ieta,iphi);
	      //	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	    if(maxDiff(caprms[0],caprms[1],caprms[2],caprms[3])>caprms_thresh_){
	      if(err_map_geo[idx])err_map_geo[idx]->Fill(ieta,iphi);
	      //	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	       
	    float avg = (capmean[0]+capmean[1]+capmean[2]+capmean[3])/4.0;
	    if(qie_mean[idx]) qie_mean[idx]->Fill(avg);
	    avg = (caprms[0]+caprms[1]+caprms[2]+caprms[3])/4.0;
	    if(qie_rms[idx]) qie_rms[idx]->Fill(avg);
	  }
	}
      }      
    }
  }

  return;
}
