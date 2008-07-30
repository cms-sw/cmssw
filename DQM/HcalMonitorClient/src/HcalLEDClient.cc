#include <DQM/HcalMonitorClient/interface/HcalLEDClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalLEDClient::HcalLEDClient(){}

void HcalLEDClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  for(int i=0; i<4; i++){    
    rms_shape_[i]=0;
    mean_shape_[i]=0;
    rms_time_[i]=0;
    mean_time_[i]=0;
    rms_energy_[i]=0;
    mean_energy_[i]=0;

    rms_shapeDep_[i]=0;
    mean_shapeDep_[i]=0;
    rms_timeDep_[i]=0;
    mean_timeDep_[i]=0;
    rms_energyDep_[i]=0;
    mean_energyDep_[i]=0;

    err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }
  HFlumi_etsum = 0;
  HFlumi_occabthr1 = 0;
  HFlumi_occbetthr1 = 0;
  HFlumi_occbelthr1 = 0;
  HFlumi_occabthr2 = 0;
  HFlumi_occbetthr2 = 0;
  HFlumi_occbelthr2 = 0;

  rms_thresh_ = ps.getUntrackedParameter<double>("LEDRMS_ErrThresh", 0.8);
  cout << "LED RMS error threshold set to " << rms_thresh_ << endl;
  
  mean_thresh_ = ps.getUntrackedParameter<double>("LEDMEAN_ErrThresh", 2.25);
  cout << "LED MEAN error threshold set to " << mean_thresh_ << endl;


  ///ntuple output file
  m_outputFileName = ps.getUntrackedParameter<string>("LEDoutputTextFile", "");
  if ( m_outputFileName.size() != 0 ) {
    cout << "Hcal LED text output will be saved to " << m_outputFileName.c_str() << endl;
    m_outTextFile.open(m_outputFileName.c_str());
    m_outTextFile<<"Det\tEta\tPhi\tD\tEnergy_Mean\tEnergy_RMS\tTime_Mean\tTime_RMS     "<<std::endl;
  }
}

HcalLEDClient::~HcalLEDClient(){
  this->cleanup();
}

void HcalLEDClient::beginJob(const EventSetup& eventSetup){
  
  if ( debug_ ) cout << "HcalLEDClient: beginJob" << endl;
  //  eventSetup.get<HcalDbRecord>().get(conditions_);
  
   // get the hcal mapping
  edm::ESHandle<HcalDbService> pSetup;
  eventSetup.get<HcalDbRecord>().get( pSetup );
  readoutMap_=pSetup->getHcalMapping();

  ievt_ = 0;
  jevt_ = 0;
  this->resetAllME();
  return;
}

void HcalLEDClient::beginRun(void){

  if ( debug_ ) cout << "HcalLEDClient: beginRun" << endl;

  jevt_ = 0;
  this->resetAllME();
  return;
}

void HcalLEDClient::endJob(void) {

  if ( debug_ ) cout << "HcalLEDClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
}

void HcalLEDClient::endRun(void) {

  if ( debug_ ) cout << "HcalLEDClient: endRun, jevt = " << jevt_ << endl;

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

      if(rms_shapeDep_[i]) delete rms_shapeDep_[i];
      if(mean_shapeDep_[i]) delete mean_shapeDep_[i];
      if(rms_timeDep_[i]) delete rms_timeDep_[i];
      if(mean_timeDep_[i]) delete mean_timeDep_[i];
      if(rms_energyDep_[i]) delete rms_energyDep_[i];
      if(mean_energyDep_[i]) delete mean_energyDep_[i];

      if(err_map_geo_[i]) delete err_map_geo_[i];
      if(err_map_elec_[i]) delete err_map_elec_[i];
      if(avg_shape_[i]) delete avg_shape_[i];
      if(avg_time_[i]) delete avg_time_[i];
      if(avg_energy_[i]) delete avg_energy_[i];
    }
      if(HFlumi_etsum) delete HFlumi_etsum;
      if(HFlumi_occabthr1) delete HFlumi_occabthr1;
      if(HFlumi_occbetthr1) delete HFlumi_occbetthr1;
      if(HFlumi_occbelthr1) delete HFlumi_occbelthr1;
      if(HFlumi_occabthr2) delete HFlumi_occabthr2;
      if(HFlumi_occbetthr2) delete HFlumi_occbetthr2;
      if(HFlumi_occbelthr2) delete HFlumi_occbelthr2;
  }

  for(int i=0; i<4; i++){    
    rms_shape_[i]=0;
    mean_shape_[i]=0;
    rms_time_[i]=0;
    mean_time_[i]=0;
    rms_energy_[i]=0;
    mean_energy_[i]=0;

    rms_shapeDep_[i]=0;
    mean_shapeDep_[i]=0;
    rms_timeDep_[i]=0;
    mean_timeDep_[i]=0;
    rms_energyDep_[i]=0;
    mean_energyDep_[i]=0;

    err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    avg_shape_[i] = 0;
    avg_time_[i] = 0;
    avg_energy_[i] = 0;
  }
  HFlumi_etsum = 0;
  HFlumi_occabthr1 = 0;
  HFlumi_occbetthr1 = 0;
  HFlumi_occbelthr1 = 0;
  HFlumi_occabthr2 = 0;
  HFlumi_occbetthr2 = 0;
  HFlumi_occbelthr2 = 0;


  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();
  return;
}

void HcalLEDClient::report(){
   if(!dbe_) return;
  if ( debug_ ) cout << "HcalLEDClient: report" << endl;

  char name[256];
  sprintf(name, "%sHcal/LEDMonitor/LED Task Event Number",process_.c_str());
  MonitorElement* me = dbe_->get(name);
  if ( me ) {
    string s = me->valueString();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
    if ( debug_ ) cout << "Found '" << name << "'" << endl;
  }
  getHistograms();
  return;
}

void HcalLEDClient::getHistograms(){
  if(!dbe_) return;
  char name[256];

  //Get mean/rms maps by Geometry
  MonitorElement* meDepTimeMean[4];
  MonitorElement* meDepTimeRMS[4];
  MonitorElement* meDepShapeMean[4];
  MonitorElement* meDepShapeRMS[4];
  MonitorElement* meDepEnergyMean[4];
  MonitorElement* meDepEnergyRMS[4];

  for(int i=0; i<4; i++){
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Time Depth %d",process_.c_str(),i+1);
    meDepTimeMean[i] = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Time Depth %d",process_.c_str(),i+1);
    meDepTimeRMS[i] = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Shape Depth %d",process_.c_str(),i+1);
    meDepShapeMean[i] = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Shape Depth %d",process_.c_str(),i+1);
    meDepShapeRMS[i] = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Energy Depth %d",process_.c_str(),i+1);
    meDepEnergyMean[i] = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Energy Depth %d",process_.c_str(),i+1);
    meDepEnergyRMS[i] = dbe_->get(name);

    if(meDepTimeMean[i]) dbe_->softReset(meDepTimeMean[i]);
    if(meDepTimeRMS[i]) dbe_->softReset(meDepTimeRMS[i]);
    if(meDepEnergyMean[i]) dbe_->softReset(meDepEnergyMean[i]);
    if(meDepEnergyRMS[i]) dbe_->softReset(meDepEnergyRMS[i]);
    if(meDepShapeMean[i]) dbe_->softReset(meDepShapeMean[i]);
    if(meDepShapeRMS[i]) dbe_->softReset(meDepShapeRMS[i]);
  }

  //Get mean/rms maps by Electronics
  MonitorElement* meFEDunpacked = NULL;
  map<unsigned int,MonitorElement*> meRMSenergyElec;
  map<unsigned int,MonitorElement*> meMEANenergyElec;
  map<unsigned int,MonitorElement*> meRMStimeElec;
  map<unsigned int,MonitorElement*> meMEANtimeElec;
  map<unsigned int,MonitorElement*> meRMSshapeElec;
  map<unsigned int,MonitorElement*> meMEANshapeElec;

  sprintf(name,"%sHcal/FEDs Unpacked",process_.c_str());
  meFEDunpacked = dbe_->get(name);
  if(meFEDunpacked){
    for(int b=1; b<=meFEDunpacked->getNbinsX(); b++){
      if(meFEDunpacked->getBinContent(b)!=0){
	
	int fedNum = b-1+700;
	sprintf(name,"%sHcal/LEDMonitor/DCC %d Mean Shape Map",process_.c_str(),fedNum);
	MonitorElement* me1 = dbe_->get(name);
	if(me1!=NULL){
	  dbe_->softReset(me1);
	  meMEANshapeElec[fedNum] = me1;	
	}
	sprintf(name,"%sHcal/LEDMonitor/DCC %d RMS Shape Map",process_.c_str(),fedNum);
	MonitorElement* me2 = dbe_->get(name);
	if(me2!=NULL){
	  dbe_->softReset(me2);
	  meRMSshapeElec[fedNum] = me2;
	}
	
	sprintf(name,"%sHcal/LEDMonitor/DCC %d Mean Energy Map",process_.c_str(),fedNum);
	MonitorElement* me3 = dbe_->get(name);
	if(me3!=NULL){
	  dbe_->softReset(me3);
	  meMEANenergyElec[fedNum] = me3;
	}
	sprintf(name,"%sHcal/LEDMonitor/DCC %d RMS Energy Map",process_.c_str(),fedNum);
	MonitorElement* me4 = dbe_->get(name);
	if(me4!=NULL){
	  dbe_->softReset(me4);
	  meRMSenergyElec[fedNum] = me4;
	}
	
	sprintf(name,"%sHcal/LEDMonitor/DCC %d Mean Time Map",process_.c_str(),fedNum);
	MonitorElement* me5 = dbe_->get(name);
	if(me5!=NULL){
	  dbe_->softReset(me5);
	  meMEANtimeElec[fedNum] = me5;	  
	}
	sprintf(name,"%sHcal/LEDMonitor/DCC %d RMS Time Map",process_.c_str(),fedNum);
	MonitorElement* me6 = dbe_->get(name);
	if(me6!=NULL){
	  dbe_->softReset(me6);
	  meRMStimeElec[fedNum] = me6;
	}
      }
    }
  }
  ///Fill histos...
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";

    sprintf(name,"LEDMonitor/%s/%s Average Pulse Shape",type.c_str(),type.c_str());      
    avg_shape_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
    sprintf(name,"LEDMonitor/%s/%s Average Pulse Time",type.c_str(),type.c_str());      
    avg_time_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
    sprintf(name,"LEDMonitor/%s/%s Average Pulse Energy",type.c_str(),type.c_str());      
    avg_energy_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
    
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Shape RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meShapeRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Shape Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meShapeMean  = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Time RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meTimeRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Time Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meTimeMean  = dbe_->get(name);

    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Energy RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meEnergyRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Energy Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meEnergyMean  = dbe_->get(name);

    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meGeoErr  = dbe_->get(name);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Elec Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meElecErr  = dbe_->get(name);

    if(i==2){ 
      sprintf(name,"LEDMonitor/%s/%s lumi ET-sum per wedge",type.c_str(),type.c_str());      
      HFlumi_etsum = getHisto(name, process_,dbe_,debug_,cloneME_);

      sprintf(name,"LEDMonitor/%s/%s lumi Occupancy above threshold ring1",type.c_str(),type.c_str());      
      HFlumi_occabthr1 = getHisto(name, process_,dbe_,debug_,cloneME_);
      sprintf(name,"LEDMonitor/%s/%s lumi Occupancy between thresholds ring1",type.c_str(),type.c_str());      
      HFlumi_occbetthr1 = getHisto(name, process_,dbe_,debug_,cloneME_);
      sprintf(name,"LEDMonitor/%s/%s lumi Occupancy below threshold ring1",type.c_str(),type.c_str());      
      HFlumi_occbelthr1 = getHisto(name, process_,dbe_,debug_,cloneME_);

      sprintf(name,"LEDMonitor/%s/%s lumi Occupancy above threshold ring2",type.c_str(),type.c_str());      
      HFlumi_occabthr2 = getHisto(name, process_,dbe_,debug_,cloneME_);
      sprintf(name,"LEDMonitor/%s/%s lumi Occupancy between thresholds ring2",type.c_str(),type.c_str());      
      HFlumi_occbetthr2 = getHisto(name, process_,dbe_,debug_,cloneME_);
      sprintf(name,"LEDMonitor/%s/%s lumi Occupancy below threshold ring2",type.c_str(),type.c_str());      
      HFlumi_occbelthr2 = getHisto(name, process_,dbe_,debug_,cloneME_);
    }

    if(!meShapeRMS || !meShapeMean) continue;
    if(!meTimeRMS || !meTimeMean) continue;
    if(!meEnergyRMS || !meEnergyMean) continue;
    dbe_->softReset(meShapeRMS); dbe_->softReset(meShapeMean);
    dbe_->softReset(meTimeRMS); dbe_->softReset(meTimeMean);
    dbe_->softReset(meEnergyRMS); dbe_->softReset(meEnergyMean);
    dbe_->softReset(meGeoErr); dbe_->softReset(meElecErr);
    
    for(int ieta=-42; ieta<=42; ieta++){
      if(ieta==0) continue;
      for(int iphi=1; iphi<=73; iphi++){
	for(int depth=1; depth<=4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  HcalSubdetector subdet = HcalBarrel;
	  if(i==1) subdet = HcalEndcap;	  
	  else if(i==2) subdet = HcalForward;
	  else if(i==3) subdet = HcalOuter;
	  HcalDetId id(subdet,ieta,iphi,depth);
	  HcalElectronicsId eid = readoutMap_->lookup(id);
	  
	  sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",
		  process_.c_str(), type.c_str(),type.c_str(),ieta,iphi,depth);  
	  MonitorElement* me = dbe_->get(name);
	  if(me){
	    meShapeRMS->Fill(me->getRMS());
	    meShapeMean->Fill(me->getMean());
	    meDepShapeRMS[depth-1]->Fill(ieta,iphi,me->getRMS());
	    meDepShapeMean[depth-1]->Fill(ieta,iphi,me->getMean());
	    
	    if(meRMSshapeElec.find(eid.dccid()+700)!=meRMSshapeElec.end()){
	      meRMSshapeElec[eid.dccid()+700]->Fill(eid.htrChanId(), eid.spigot(), me->getRMS());
	      meMEANshapeElec[eid.dccid()+700]->Fill(eid.htrChanId(), eid.spigot(), me->getMean());
	    }
	    //else printf("HcalLEDClient:  we should have had a histo for DCC %d!!\n",eid.dccid()+700);
	    

	    if(me->getRMS()<rms_thresh_ || me->getMean()>mean_thresh_){
	      meGeoErr->Fill(ieta,iphi);
	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());	      
	    }
	  }
	  

	  float timeMeanVal = -1; float enMeanVal = -1;
	  float timeRMSVal = -1; float enRMSVal = -1;
	  sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",process_.c_str(),
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = dbe_->get(name);
	  if(me){
	    timeMeanVal = me->getMean();
	    timeRMSVal = me->getRMS();
	    meTimeRMS->Fill(timeRMSVal);
	    meTimeMean->Fill(timeMeanVal);	
	    meDepTimeRMS[depth-1]->Fill(ieta,iphi,timeRMSVal);
	    meDepTimeMean[depth-1]->Fill(ieta,iphi,timeMeanVal);
	    if(meRMStimeElec.find(eid.dccid()+700)!=meRMStimeElec.end()){
	      meRMStimeElec[eid.dccid()+700]->Fill(eid.htrChanId(), eid.spigot(), timeRMSVal);
	      meMEANtimeElec[eid.dccid()+700]->Fill(eid.htrChanId(), eid.spigot(), timeMeanVal);
	    }
	    //else{
	    // printf("HcalLEDClient:  we should had had a histo for DCC %d!!\n",eid.dccid()+700);
	    //}
	  }

	  sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",process_.c_str(),
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  me = dbe_->get(name);
	  if(me){
	    enMeanVal = me->getMean();
	    enRMSVal = me->getRMS();
	    meEnergyRMS->Fill(enRMSVal);
	    meEnergyMean->Fill(enMeanVal);
	    meDepEnergyRMS[depth-1]->Fill(ieta,iphi,enRMSVal);
	    meDepEnergyMean[depth-1]->Fill(ieta,iphi,enMeanVal);
	    if(meRMSenergyElec.find(eid.dccid()+700)!=meRMSenergyElec.end()){
	      meRMSenergyElec[eid.dccid()+700]->Fill(eid.htrChanId(), eid.spigot(), enRMSVal);
	      meMEANenergyElec[eid.dccid()+700]->Fill(eid.htrChanId(), eid.spigot(), enMeanVal);
	    }	    
	    //else{
	    //  printf("HcalLEDClient:  we should had had a histo for DCC %d!!\n",eid.dccid()+700);
	    //}

	    if(depth==1 || depth==2)
	      m_outTextFile<<"HF\t"<<ieta<<"\t"<<iphi<<"\t"<<depth<<"\t"<<enMeanVal<<"\t"<<"\t"<<enRMSVal<<"\t"<<"\t"<<timeMeanVal<<"\t"<<"\t"<<timeRMSVal<<std::endl;

	  }	  
	}
      }
    }
    
    rms_shape_[i]=getHisto(meShapeRMS,debug_,cloneME_);
    mean_shape_[i]=getHisto(meShapeMean,debug_,cloneME_);
    rms_time_[i]=getHisto(meTimeRMS,debug_,cloneME_);
    mean_time_[i]=getHisto(meTimeMean,debug_,cloneME_);
    rms_energy_[i]=getHisto(meEnergyRMS,debug_,cloneME_);
    mean_energy_[i]=getHisto(meEnergyMean,debug_,cloneME_);

    
    err_map_geo_[i]=getHisto2(meGeoErr,debug_,cloneME_);
    err_map_elec_[i]=getHisto2(meElecErr,debug_,cloneME_);
  }

  for(int i=0; i<4; i++){
    rms_shapeDep_[i]=getHisto2(meDepShapeRMS[i],debug_,cloneME_);
    mean_shapeDep_[i]=getHisto2(meDepShapeMean[i],debug_,cloneME_);
    rms_timeDep_[i]=getHisto2(meDepTimeRMS[i],debug_,cloneME_);
    mean_timeDep_[i]=getHisto2(meDepTimeMean[i],debug_,cloneME_);
    rms_energyDep_[i]=getHisto2(meDepEnergyRMS[i],debug_,cloneME_);
    mean_energyDep_[i]=getHisto2(meDepEnergyMean[i],debug_,cloneME_);
  }

  for(map<unsigned int, MonitorElement*>::iterator meIter = meRMStimeElec.begin();
      meIter !=meRMStimeElec.end();
      meIter ++){

    rms_timeElec_[meIter->first] = getHisto2(meIter->second,debug_,cloneME_);
    mean_timeElec_[meIter->first] = getHisto2(meMEANtimeElec[meIter->first],debug_,cloneME_);
 
    rms_energyElec_[meIter->first] = getHisto2(meRMSenergyElec[meIter->first],debug_,cloneME_);
    mean_energyElec_[meIter->first] = getHisto2(meMEANenergyElec[meIter->first],debug_,cloneME_);

    rms_shapeElec_[meIter->first] = getHisto2(meRMSshapeElec[meIter->first],debug_,cloneME_);
    mean_shapeElec_[meIter->first] = getHisto2(meMEANshapeElec[meIter->first],debug_,cloneME_);
  }
  
  return;
}

void HcalLEDClient::analyze(void){
  
  jevt_++;
  int updates = 0;
  if ( (updates % 10) == 0 ) {
    if ( debug_ ) cout << "HcalLEDClient: " << updates << " updates" << endl;
  }
  getHistograms();
  return;
}

void HcalLEDClient::createTests(){
  if(!dbe_) return;
  
  char meTitle[250], name[250];    
  vector<string> params;

  if(debug_) printf("Creating LED tests...\n");

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    if(i==2) type = "HF";
    if(i==3) type = "HO";

    sprintf(meTitle,"%sHcal/LEDMonitor/%s/%s LED Shape RMS Values",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s LED Shape RMS Values: X-Range",type.c_str());
    if( dqmQtests_.find(name) == dqmQtests_.end() ){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	char low[20];
	sprintf(low,"%.2f\n",rms_thresh_);
	params.push_back(low); params.push_back("2");  //xmin, xmax
	createXRangeTest(dbe_, params);
      }
    }
    
    sprintf(meTitle,"%sHcal/LEDMonitor/%s/%s LED Shape Mean Values",process_.c_str(),type.c_str(),type.c_str());
    sprintf(name,"%s LED Shape RMS Values: X-Range",type.c_str());
    if( dqmQtests_.find(name) == dqmQtests_.end() ){	
      MonitorElement* me = dbe_->get(meTitle);
      if(me){
	dqmQtests_[name]=meTitle;	  
	params.clear();
	params.push_back(meTitle); params.push_back(name);  //hist and test titles
	params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	char high[20];
	sprintf(high,"%.2f\n",mean_thresh_);
	params.push_back("0"); params.push_back(high);  //xmin, xmax
	createXRangeTest(dbe_, params);
      }
    }    
  }

  return;
}

void HcalLEDClient::resetAllME(){
  if(!dbe_) return;
  Char_t name[150];    

  for(int i=1; i<5; i++){
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Time Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Time Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Shape Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Shape Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Energy Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Energy Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
  }


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO"; 

    sprintf(name,"%sHcal/LEDMonitor/%s/%s Ped Subtracted Pulse Shape",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s Average Pulse Shape",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Shape RMS Values",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Shape Mean Values",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s Average Pulse Time",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Time RMS Values",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Time Mean Values",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s Average Pulse Energy",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Energy RMS Values",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Energy Mean Values",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Elec Error Map",process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    
    for(int ieta=-42; ieta<42; ieta++){
      if(ieta==0) continue;
      for(int iphi=0; iphi<73; iphi++){
	for(int depth=1; depth<4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",
		  process_.c_str(), type.c_str(),type.c_str(),ieta,iphi,depth);  
	  resetME(name,dbe_);
	  sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",
		  process_.c_str(),type.c_str(),type.c_str(),ieta,iphi,depth);  
	  resetME(name,dbe_);
	  sprintf(name,"%sHcal/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",
		  process_.c_str(),type.c_str(),type.c_str(),ieta,iphi,depth);  
	  resetME(name,dbe_);
	}
      }
    }
  }
  return;
}

void HcalLEDClient::htmlOutput(int runNo, string htmlDir, string htmlName){
  
  cout << "Preparing HcalLEDClient html output ..." << endl;
  string client = "LEDMonitor";
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
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
  htmlFile << "<a href=\"#GEO_Plots\">Geometry Plots </a></br>" << endl;
  htmlFile << "<a href=\"#ELEC_Plots\">Electronics Plots </a></br>" << endl;
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
  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"GEO_Plots\"><h3>Geometry Histograms</h3></td></tr>" << endl;
  histoHTML2(runNo,mean_timeDep_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_timeDep_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_timeDep_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_timeDep_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_timeDep_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_timeDep_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_timeDep_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_timeDep_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_energyDep_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_energyDep_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_energyDep_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_energyDep_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_energyDep_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_energyDep_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_energyDep_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_energyDep_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_shapeDep_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_shapeDep_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_shapeDep_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_shapeDep_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_shapeDep_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_shapeDep_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,mean_shapeDep_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,rms_shapeDep_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\"ELEC_Plots\"><h3>Electronics Histograms</h3></td></tr>" << endl;
  for(map<unsigned int, TH2F*>::iterator hIter = rms_energyElec_.begin();
      hIter!=rms_energyElec_.end();
      hIter++){
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,rms_timeElec_[hIter->first],"HTR Channel","Spigot", 92, htmlFile,htmlDir);
    histoHTML2(runNo,mean_timeElec_[hIter->first],"HTR Channel","Spigot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,rms_energyElec_[hIter->first],"HTR Channel","Spigot", 92, htmlFile,htmlDir);
    histoHTML2(runNo,mean_energyElec_[hIter->first],"HTR Channel","Spigot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML2(runNo,rms_shapeElec_[hIter->first],"HTR Channel","Spigot", 92, htmlFile,htmlDir);
    histoHTML2(runNo,mean_shapeElec_[hIter->first],"HTR Channel","Spigot", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

  }
  htmlFile << "</tr>" << endl;

   for(int i=0; i<4; i++){
     if(!subDetsOn_[i]) continue; 
     string type = "HB";
     if(i==1) type = "HE"; 
     else if(i==2) type = "HF"; 
     else if(i==3) type = "HO"; 
     
     htmlFile << "<tr align=\"left\">" << endl;  
     htmlFile << "<td>&nbsp;&nbsp;&nbsp;<a name=\""<<type<<"_Plots\"><h3>" << type << " Histograms</h3></td></tr>" << endl;
     
     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML2(runNo,err_map_geo_[i],"iEta","iPhi", 92, htmlFile,htmlDir);
     histoHTML2(runNo,err_map_elec_[i],"VME Crate ID","HTR Slot", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;
     
     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(runNo,avg_shape_[i],"Average Pulse Shape","Events", 92, htmlFile,htmlDir);
     histoHTML(runNo,avg_time_[i],"Average Pulse Time","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;
     
     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(runNo,avg_energy_[i],"Average ADC Sum","Events", 92, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(runNo,rms_shape_[i],"Shape RMS Value","Events", 92, htmlFile,htmlDir);
     histoHTML(runNo,mean_shape_[i],"Shape Mean Value","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(runNo,rms_time_[i],"Time RMS Value","Events", 92, htmlFile,htmlDir);
     histoHTML(runNo,mean_time_[i],"Time Mean Value","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     htmlFile << "<tr align=\"left\">" << endl;
     histoHTML(runNo,rms_energy_[i],"ADC Sum RMS Value","Events", 92, htmlFile,htmlDir);
     histoHTML(runNo,mean_energy_[i],"ADC Sum Mean Value","Events", 100, htmlFile,htmlDir);
     htmlFile << "</tr>" << endl;

     if(i==2){
      htmlFile << "<tr align=\"left\">" << endl;
      histoHTML(runNo,HFlumi_etsum,"Wedge number","ET Sum times events", 92, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;

      htmlFile << "<tr align=\"left\">" << endl;
      histoHTML(runNo,HFlumi_occabthr1,"Wedge number","Occupancy times events", 100, htmlFile,htmlDir);
      histoHTML(runNo,HFlumi_occabthr2,"Wedge number","Occupancy times events", 100, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;

      htmlFile << "<tr align=\"left\">" << endl;
      histoHTML(runNo,HFlumi_occbetthr1,"Wedge number","Occupancy times events", 100, htmlFile,htmlDir);
      histoHTML(runNo,HFlumi_occbetthr2,"Wedge number","Occupancy times events", 100, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;

      htmlFile << "<tr align=\"left\">" << endl;
      histoHTML(runNo,HFlumi_occbelthr1,"Wedge number","Occupancy times events", 100, htmlFile,htmlDir);
      histoHTML(runNo,HFlumi_occbelthr2,"Wedge number","Occupancy times events", 100, htmlFile,htmlDir);
      htmlFile << "</tr>" << endl;
     }

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

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/LEDMonitor/LED Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }
  char name[256];
  for(int i=0; i<4; i++){
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Time Depth %d",process_.c_str(),i+1);
    mean_timeDep_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Time Depth %d",process_.c_str(),i+1);
    rms_timeDep_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Shape Depth %d",process_.c_str(),i+1);
    mean_shapeDep_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Shape Depth %d",process_.c_str(),i+1);
    rms_shapeDep_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED Mean Energy Depth %d",process_.c_str(),i+1);
    mean_energyDep_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"%sHcal/LEDMonitor/LED RMS Energy Depth %d",process_.c_str(),i+1);
    rms_energyDep_[i]=(TH2F*)infile->Get(name);
  }


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue; 
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO";


    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s Average Pulse Shape",type.c_str(),type.c_str());      
    avg_shape_[i] = (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s Average Pulse Time",type.c_str(),type.c_str());      
    avg_time_[i] = (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s Average Pulse Energy",type.c_str(),type.c_str());      
    avg_energy_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Shape RMS Values",type.c_str(),type.c_str());
    rms_shape_[i]=(TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Shape Mean Values",type.c_str(),type.c_str());
    mean_shape_[i]=(TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Time RMS Values",type.c_str(),type.c_str());
    rms_time_[i]=(TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Time Mean Values",type.c_str(),type.c_str());
    mean_time_[i]=(TH1F*)infile->Get(name);

    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Energy RMS Values",type.c_str(),type.c_str());
    rms_energy_[i]=(TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Energy Mean Values",type.c_str(),type.c_str());
    mean_energy_[i]=(TH1F*)infile->Get(name);


    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Geo Error Map",type.c_str(),type.c_str());
    err_map_geo_[i]=(TH2F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Elec Error Map",type.c_str(),type.c_str());
    err_map_elec_[i]=(TH2F*)infile->Get(name);

    if(i==2){
     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi ET-sum per wedge",type.c_str(),type.c_str());      
     HFlumi_etsum = (TH1F*)infile->Get(name);

     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi Occupancy above threshold ring1",type.c_str(),type.c_str()); 
     HFlumi_occabthr1 = (TH1F*)infile->Get(name);
     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi Occupancy between thresholds ring1",type.c_str(),type.c_str()); 
     HFlumi_occbetthr1 = (TH1F*)infile->Get(name);
     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi Occupancy below threshold ring1",type.c_str(),type.c_str()); 
     HFlumi_occbelthr1 = (TH1F*)infile->Get(name);


     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi Occupancy above threshold ring2",type.c_str(),type.c_str()); 
     HFlumi_occabthr2 = (TH1F*)infile->Get(name);
     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi Occupancy between thresholds ring2",type.c_str(),type.c_str()); 
     HFlumi_occbetthr2 = (TH1F*)infile->Get(name);
     sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s lumi Occupancy below threshold ring2",type.c_str(),type.c_str()); 
     HFlumi_occbelthr2 = (TH1F*)infile->Get(name);
    }

    for(int ieta=-42; ieta<=42; ieta++){
      if(ieta==0) continue;
      for(int iphi=1; iphi<=73; iphi++){
	for(int depth=1; depth<=4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Shape ieta=%d iphi=%d depth=%d",
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  TH1F* h = (TH1F*)infile->Get(name);
	  if(h){
	    rms_shape_[i]->Fill(h->GetRMS());
	    mean_shape_[i]->Fill(h->GetMean());
	  }
	  
	  sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Time ieta=%d iphi=%d depth=%d",
		  type.c_str(),type.c_str(),ieta,iphi,depth);  
	  h = (TH1F*)infile->Get(name);
	  if(h){
	    rms_time_[i]->Fill(h->GetRMS());
	    mean_time_[i]->Fill(h->GetMean());
	  }	  

	  sprintf(name,"DQMData/Hcal/LEDMonitor/%s/%s LED Energy ieta=%d iphi=%d depth=%d",
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

