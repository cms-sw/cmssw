#include <DQM/HcalMonitorClient/interface/HcalPedestalClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>

HcalPedestalClient::HcalPedestalClient(){}

void HcalPedestalClient::init(const ParameterSet& ps, DQMStore* dbe,string clientName){
  //Call the base class first
  HcalBaseClient::init(ps,dbe,clientName);

  readoutMap_=0;

  for(int i=0; i<4; i++){
    all_peds_[i]=0;   ped_rms_[i]=0;
    ped_mean_[i]=0;   capid_rms_[i]=0;
    sub_mean_[i]=0;   sub_rms_[i]=0;
    capid_mean_[i]=0; qie_rms_[i]=0;
    qie_mean_[i]=0;   err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    pedMapMean_E[i] = 0;
    pedMapRMS_E[i] = 0;
    pedMapMeanD_[i] = 0;
    pedMapRMSD_[i] = 0;
  }


  // per channel tests switch
  doPerChanTests_ = ps.getUntrackedParameter<bool>("DoPerChanTests", false);
  if(doPerChanTests_) 
    cout << "Will perform per-channel pedestal tests." << endl;

  // plot peds in RAW or subtracted values in iEta/iPhi maps
  plotPedRAW_ = ps.getUntrackedParameter<bool>("plotPedRAW", false);
  if(plotPedRAW_) 
    cout << "Plotting pedestals in RAW format." << endl;
  else
    cout << "Plotting pedestals in subtracted format." << endl;

  pedrms_thresh_ = ps.getUntrackedParameter<double>("PedestalRMS_ErrThresh", 1);
  cout << "Pedestal RMS error threshold set to " << pedrms_thresh_ << endl;

  pedmean_thresh_ = ps.getUntrackedParameter<double>("PedestalMEAN_ErrThresh", 2);
  cout << "Pedestal MEAN error threshold set to " << pedmean_thresh_ << endl;

  caprms_thresh_ = ps.getUntrackedParameter<double>("CapIdRMS_ErrThresh", 0.25);
  cout << "CapId RMS Variance error threshold set to " << caprms_thresh_ << endl;

  capmean_thresh_ = ps.getUntrackedParameter<double>("CapIdMEAN_ErrThresh", 1.5);
  cout << "CapId MEAN Variance error threshold set to " << capmean_thresh_ << endl;

}

HcalPedestalClient::~HcalPedestalClient(){
  this->cleanup();
}

void HcalPedestalClient::beginJob(const EventSetup& eventSetup){

  if ( debug_ ) cout << "HcalPedestalClient: beginJob" << endl;
  eventSetup.get<HcalDbRecord>().get(conditions_);

  // get the hcal mapping
  edm::ESHandle<HcalDbService> pSetup;
  eventSetup.get<HcalDbRecord>().get( pSetup );
  readoutMap_=pSetup->getHcalMapping();

  ievt_ = 0;
  jevt_ = 0;
  this->setup();
  return;
}

void HcalPedestalClient::beginRun(void){

  if ( debug_ ) cout << "HcalPedestalClient: beginRun" << endl;

  jevt_ = 0;
  this->setup();
  this->resetAllME();
  return;
}

void HcalPedestalClient::endJob(void) {

  if ( debug_ ) cout << "HcalPedestalClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();
  return;
}

void HcalPedestalClient::endRun(void) {

  if ( debug_ ) cout << "HcalPedestalClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();
  return;
}

void HcalPedestalClient::setup(void) {

  return;
}

void HcalPedestalClient::cleanup(void) {
  if(cloneME_){
    for(int i=0; i<4; i++){
      if(all_peds_[i]);     delete all_peds_[i];   
      if(ped_rms_[i]);      delete ped_rms_[i];
      if(ped_mean_[i]);     delete ped_mean_[i];   

      if(sub_rms_[i]);      delete sub_rms_[i];
      if(sub_mean_[i]);     delete sub_mean_[i];   
      if(capid_rms_[i]);    delete capid_rms_[i];
      if(capid_mean_[i]);   delete capid_mean_[i]; 
      if(qie_rms_[i]);      delete qie_rms_[i];
      if(qie_mean_[i]);     delete qie_mean_[i];   
      if(err_map_geo_[i]);  delete err_map_geo_[i];
      if(err_map_elec_[i]); delete err_map_elec_[i];
      if(pedMapMean_E[i]);  delete pedMapMean_E[i];
      if(pedMapRMS_E[i]);   delete pedMapRMS_E[i];
      if(pedMapMeanD_[i]);  delete pedMapMeanD_[i];
      if(pedMapRMSD_[i]);   delete pedMapRMSD_[i];
    }
  }
  for(int i=0; i<4; i++){
    all_peds_[i]=0;   ped_rms_[i]=0;
    ped_mean_[i]=0;   capid_rms_[i]=0;
    sub_mean_[i]=0;   sub_rms_[i]=0;
    capid_mean_[i]=0; qie_rms_[i]=0;
    qie_mean_[i]=0;   err_map_geo_[i]=0;
    err_map_elec_[i]=0;
    pedMapMean_E[i] = 0;
    pedMapRMS_E[i] = 0;
    pedMapMeanD_[i] = 0;
    pedMapRMSD_[i] = 0;
  }

  dqmReportMapErr_.clear(); dqmReportMapWarn_.clear(); dqmReportMapOther_.clear();
  dqmQtests_.clear();

  return;
}

void HcalPedestalClient::report(){
   if(!dbe_) return;
  if ( debug_ ) cout << "HcalPedestalClient: report" << endl;
  this->setup();

  char name[256];    
  sprintf(name, "%sHcal/PedestalMonitor/Pedestal Task Event Number",process_.c_str());
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

void HcalPedestalClient::getHistograms(){
   if(!dbe_) return;

   char name[256];    

  MonitorElement* meMeanMap_D[4];
  MonitorElement* meRMSMap_D[4];
  MonitorElement* meMeanMap_E[3];
  MonitorElement* meRMSMap_E[3];

  for(int i=0; i<4; i++){
    sprintf(name,"%sHcal/PedestalMonitor/Ped Mean Depth %d",process_.c_str(),i+1);
    meMeanMap_D[i]  = dbe_->get(name);
    sprintf(name,"%sHcal/PedestalMonitor/Ped RMS Depth %d",process_.c_str(),i+1);
    meRMSMap_D[i]  = dbe_->get(name);
  }
  
  sprintf(name,"%sHcal/PedestalMonitor/Ped Mean by Crate-Slot",process_.c_str());
  meMeanMap_E[0] = dbe_->get(name);
  sprintf(name,"%sHcal/PedestalMonitor/Ped RMS by Crate-Slot",process_.c_str());
  meRMSMap_E[0] = dbe_->get(name);
  
  sprintf(name,"%sHcal/PedestalMonitor/Ped Mean by Fiber-Chan",process_.c_str());
  meMeanMap_E[1] = dbe_->get(name);
  sprintf(name,"%sHcal/PedestalMonitor/Ped RMS by Fiber-Chan",process_.c_str());
  meRMSMap_E[1] = dbe_->get(name);


  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";
    
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* mePedRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* mePedMean = dbe_->get(name);

    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Normalized RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meSubRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Subtracted Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meSubMean = dbe_->get(name);


    sprintf(name,"%sHcal/PedestalMonitor/%s/%s CapID RMS Variance",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meCapRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s CapID Mean Variance",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meCapMean = dbe_->get(name);
    
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s QIE RMS Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meQieRMS  = dbe_->get(name);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s QIE Mean Values",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meQieMean = dbe_->get(name);
    
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meGeoErr  = dbe_->get(name);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Elec Error Map",process_.c_str(),type.c_str(),type.c_str());
    MonitorElement* meElecErr = dbe_->get(name);
    
    if(!mePedRMS || !mePedMean)continue;
    if(!meSubRMS || !meSubMean)continue;
    if(!meCapRMS || !meCapMean)continue;
    if(!meQieRMS || !meQieMean)continue;
    
    if(false) { //dbe_){      
      dbe_->softReset(mePedRMS); dbe_->softReset(mePedMean);
      dbe_->softReset(meSubRMS); dbe_->softReset(meSubMean);
      dbe_->softReset(meCapRMS); dbe_->softReset(meCapMean);
      dbe_->softReset(meQieRMS); dbe_->softReset(meQieMean);
      dbe_->softReset(meGeoErr); dbe_->softReset(meElecErr);
      dbe_->softReset(meMeanMap_D[0]); dbe_->softReset(meRMSMap_D[0]);
      dbe_->softReset(meMeanMap_D[1]); dbe_->softReset(meRMSMap_D[1]);
      dbe_->softReset(meMeanMap_D[2]); dbe_->softReset(meRMSMap_D[2]);
      dbe_->softReset(meMeanMap_D[3]); dbe_->softReset(meRMSMap_D[3]);

      dbe_->softReset(meMeanMap_E[0]); dbe_->softReset(meRMSMap_E[0]);
      dbe_->softReset(meMeanMap_E[1]); dbe_->softReset(meRMSMap_E[1]);
    }
    bool capidOK = false;
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

	  float capmeanS[4]; float caprmsS[4];	  
	  float capmeanP[4]; float caprmsP[4];	  
	  capidOK = true;
	  for(int capid=0; capid<4 && capidOK; capid++){
	    capmeanP[capid]=0; caprmsP[capid]=0; 
	    capmeanS[capid]=0; caprmsS[capid]=0; 
	    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Value (ADC) ieta=%d iphi=%d depth=%d CAPID=%d",process_.c_str(),
		    type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    MonitorElement* mePed = dbe_->get(name);
	    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Value (Subtracted) ieta=%d iphi=%d depth=%d CAPID=%d",process_.c_str(),
		    type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    MonitorElement* meSub = dbe_->get(name);

	    if(mePed!=NULL){
	      if(mePed->getEntries()>0 || true){
		capmeanP[capid] = mePed->getMean();
		caprmsP[capid] = mePed->getRMS();

		capmeanS[capid] = meSub->getMean();
		caprmsS[capid] = meSub->getRMS();

		mePedMean->Fill(capmeanP[capid]);
		mePedRMS->Fill(caprmsP[capid]);


		double width=1.0;
		if(readoutMap_){
		  const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
		  if(pedw) width = pedw->getWidth(capid);
		  if(width>0) meSubRMS->Fill(caprmsS[capid]/width);
		}
		meSubMean->Fill(capmeanS[capid]);

		if(meSub->getRMS()>pedrms_thresh_ || fabs(meSub->getMean())>pedmean_thresh_) {
		  meGeoErr->Fill(ieta,iphi);
		  meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
		}
	      }
	      else capidOK=false;	
	    }
	    else capidOK=false;	    
	  }
	  float avgMean = 100; float avgRMS=100;
	  if(capidOK){
	    meCapMean->Fill(maxDiff(capmeanP[0],capmeanP[1],capmeanP[2],capmeanP[3]));
	    meCapRMS->Fill(maxDiff(caprmsP[0],caprmsP[1],caprmsP[2],caprmsP[3]));
	    
	    if(maxDiff(capmeanP[0],capmeanP[1],capmeanP[2],capmeanP[3])>capmean_thresh_){
	      meGeoErr->Fill(ieta,iphi);
	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	    if(maxDiff(caprmsP[0],caprmsP[1],caprmsP[2],caprmsP[3])>caprms_thresh_){
	      meGeoErr->Fill(ieta,iphi);
	      meElecErr->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	       
	    float avg = (capmeanP[0]+capmeanP[1]+capmeanP[2]+capmeanP[3])/4.0;
	    meQieMean->Fill(avg);
	    avg = (caprmsP[0]+caprmsP[1]+caprmsP[2]+caprmsP[3])/4.0;
	    meQieRMS->Fill(avg);

	    avgMean = (capmeanS[0]+capmeanS[1]+capmeanS[2]+capmeanS[3])/4.0;
	    avgRMS = (caprmsS[0]+caprmsS[1]+caprmsS[2]+caprmsS[3])/4.0;
	    if(plotPedRAW_){
	      avgMean = (capmeanP[0]+capmeanP[1]+capmeanP[2]+capmeanP[3])/4.0;
	      avgRMS = (caprmsP[0]+caprmsP[1]+caprmsP[2]+caprmsP[3])/4.0;
	    }
	  }
	  if(avgMean!=100 && depth>0){
	    if(meMeanMap_D[depth-1]!=0) meMeanMap_D[depth-1]->Fill(ieta,iphi,avgMean);
	    if(meRMSMap_D[depth-1]!=0) meRMSMap_D[depth-1]->Fill(ieta,iphi,avgRMS);
	    if(meMeanMap_E[0]!=0) meMeanMap_E[0]->Fill(eid.readoutVMECrateId(),eid.htrSlot(),avgMean);
	    if(meMeanMap_E[1]!=0) meMeanMap_E[1]->Fill(eid.fiberChanId(),eid.fiberIndex(),avgMean);
	    if(meRMSMap_E[0]!=0) meRMSMap_E[0]->Fill(eid.readoutVMECrateId(),eid.htrSlot(),avgRMS);
	    if(meRMSMap_E[1]!=0) meRMSMap_E[1]->Fill(eid.fiberChanId(),eid.fiberIndex(),avgRMS);
	  }
	}
      }      
    }

    sprintf(name,"PedestalMonitor/%s/%s All Pedestal Values",type.c_str(),type.c_str());      
    all_peds_[i] = getHisto(name, process_,dbe_,debug_,cloneME_);
    
    ped_rms_[i] = getHisto(mePedRMS,debug_,cloneME_);
    ped_mean_[i] = getHisto(mePedMean,debug_,cloneME_);

    sub_rms_[i] = getHisto(meSubRMS,debug_,cloneME_);
    sub_mean_[i] = getHisto(meSubMean,debug_,cloneME_);
    
    capid_rms_[i] = getHisto(meCapRMS,debug_,cloneME_);
    capid_mean_[i] = getHisto(meCapMean,debug_,cloneME_);
    
    qie_rms_[i] = getHisto(meQieRMS,debug_,cloneME_);
    qie_mean_[i] = getHisto(meQieMean,debug_,cloneME_);
    
    err_map_geo_[i] = getHisto2(meGeoErr,debug_,cloneME_);
    err_map_elec_[i] = getHisto2(meElecErr,debug_,cloneME_);
    
  }

  for(int i=0; i<4; i++){
    pedMapMeanD_[i] = getHisto2(meMeanMap_D[i],debug_,cloneME_);
    pedMapRMSD_[i] = getHisto2(meRMSMap_D[i],debug_,cloneME_);
  }
  
  pedMapMean_E[0] = getHisto2(meMeanMap_E[0],debug_,cloneME_);
  pedMapRMS_E[0] = getHisto2(meRMSMap_E[0],debug_,cloneME_);

  pedMapMean_E[1] = getHisto2(meMeanMap_E[1],debug_,cloneME_);
  pedMapRMS_E[1] = getHisto2(meRMSMap_E[1],debug_,cloneME_);

  return;
}

void HcalPedestalClient::analyze(void){

  jevt_++;
  int updates = 0;
  if ( updates % 10 == 0 ) {
    if ( debug_ ) cout << "HcalPedestalClient: " << updates << " updates" << endl;
  }
  getHistograms();
  return;
}

void HcalPedestalClient::createTests(){
  if(!dbe_) return;
  
  char meTitle[250], name[250];    
  vector<string> params;
  
  if(debug_) printf("Creating Pedestal tests...\n");
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";
    
    if(i<3){
      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s Pedestal Geo Error Map",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s Pedestal Errors by Geometry",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = dbe_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back((string)meTitle); params.push_back((string)name);  //hist and qtest titles
	  params.push_back("0"); params.push_back("1e-10");  //mean ranges
	  params.push_back("0"); params.push_back("1e-10");  //rms ranges
	  createH2ContentTest(dbe_, params);
	}
      }
      
      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s All Pedestal Values",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s All Pedestal Values: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = dbe_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	  params.push_back("0"); params.push_back("10");  //xmin, xmax
	  createXRangeTest(dbe_, params);
	}
      }

      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s Pedestal RMS Values",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s Pedestal RMS Values: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = dbe_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	  char high[20];
	  sprintf(high,"%f",pedrms_thresh_);
	  params.push_back("0"); params.push_back(high);  //xmin, xmax
	  createXRangeTest(dbe_, params);
	}
      }
      
      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s Pedestal Mean Values",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s Pedestal Mean Values: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = dbe_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("0.75"); params.push_back("0.5");  //warn, err probs
	  char low[20];
	  sprintf(low,"%f",pedmean_thresh_);
	  params.push_back(low); params.push_back("9");  //xmin, xmax
	  createXRangeTest(dbe_, params);
	}
      }

      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s CapID RMS Variance",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s CapId RMS Variance: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = dbe_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	  char high[20];
	  sprintf(high,"%f",caprms_thresh_);
	  params.push_back("0"); params.push_back(high);  //xmin, xmax
	  createXRangeTest(dbe_, params);
	}
      }
      
      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s CapID Mean Variance",process_.c_str(),type.c_str(),type.c_str());
      sprintf(name,"%s CapId Mean Variance: X-Range",type.c_str());
      if( dqmQtests_.find(name) == dqmQtests_.end() ){	
	MonitorElement* me = dbe_->get(meTitle);
	if(me){
	  dqmQtests_[name]=meTitle;	  
	  params.clear();
	  params.push_back(meTitle); params.push_back(name);  //hist and test titles
	  params.push_back("1.0"); params.push_back("0.95");  //warn, err probs
	  char high[20];
	  sprintf(high,"%f",capmean_thresh_);
	  params.push_back("0"); params.push_back(high);  //xmin, xmax
	  createXRangeTest(dbe_, params);
	}
      }

    }
    
    if(doPerChanTests_){
      for(int ieta=-42; ieta<=42; ieta++){
	if(ieta==0) continue;
	for(int iphi=1; iphi<=73; iphi++){
	  for(int depth=1; depth<=4; depth++){
	    if(!isValidGeom(i,ieta,iphi,depth)) continue;	    

	    HcalSubdetector subdet = HcalBarrel;
	    if(i==1) subdet = HcalEndcap;
	    else if(i==2) subdet = HcalForward;
	    else if(i==3) subdet = HcalOuter;
	    const HcalDetId id(subdet,ieta,iphi,depth);
	    bool qie = true;
	    for(int capid=0; capid<4 && qie; capid++){
	      sprintf(meTitle,"%sHcal/PedestalMonitor/%s/%s Pedestal Value (ADC) ieta=%d iphi=%d depth=%d CAPID=%d", process_.c_str(),type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	      sprintf(name,"%s Pedestal ieta=%d iphi=%d depth=%d CAPID=%d: Sigma",type.c_str(),ieta,iphi,depth,capid);  
	      if( dqmQtests_.find(name) == dqmQtests_.end()){ 
		string test = ((string)name);
		MonitorElement* me = dbe_->get(meTitle);
		if(me!=NULL){
		  dqmQtests_[name]=meTitle;			
		  float mean = 0; float width = 1;
		  const HcalPedestal* pedm = (*conditions_).getPedestal(id);
		  if(pedm) mean = pedm->getValue(capid);
		  const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
		  if(pedw) width = pedw->getWidth(capid);
		  
		  params.clear();
		  params.push_back(meTitle); params.push_back(name);  //hist and test titles
		  params.push_back("0.317"); params.push_back("0.0455");  //warn, err prob
		  //		  params.push_back("1.0"); params.push_back("0");  //warn, err prob
		  char m[20]; sprintf(m,"%f",mean);
		  char w[20]; sprintf(w,"%f",width);
		  params.push_back(m);  params.push_back(w);  //mean, sigma
		  params.push_back("useSigma");  // useSigma or useRMS
		  createMeanValueTest(dbe_, params);
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

void HcalPedestalClient::resetAllME(){
  if(!dbe_) return;
  Char_t name[150];    

  for(int i=1; i<5; i++){
    sprintf(name,"%sHcal/PedestalMonitor/Ped Mean Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Ped RMS Depth %d",process_.c_str(),i);
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Ped Mean by Crate-Slot",process_.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Ped RMS by Crate-Slot",process_.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Ped Mean by Fiber-Chan",process_.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Ped RMS by Fiber-Chan",process_.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Pedestal Mean Reference Values",process_.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/Pedestal RMS Reference Values",process_.c_str());
    resetME(name,dbe_);
  }

  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;    
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF"; 
    else if(i==3) type = "HO"; 
    
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s All Pedestal Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal RMS Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Mean Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Normalized RMS Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Subtracted Mean Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s CapID RMS Variance",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s CapID Mean Variance",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s QIE RMS Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s QIE Mean Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Geo Error Map",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Elec Error Map",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Mean Reference Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal RMS Reference Values",
	    process_.c_str(),type.c_str(),type.c_str());
    resetME(name,dbe_);
    
    for(int ieta=-42; ieta<42; ieta++){
      for(int iphi=0; iphi<72; iphi++){
	for(int depth=0; depth<4; depth++){
	  for(int capid=0; capid<4; capid++){
	    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Value (ADC) ieta=%d iphi=%d depth=%d CAPID=%d",process_.c_str(), type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    resetME(name,dbe_);
	    sprintf(name,"%sHcal/PedestalMonitor/%s/%s Pedestal Value (Subtracted) ieta=%d iphi=%d depth=%d CAPID=%d",process_.c_str(), type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    resetME(name,dbe_);
	  }
	}
      }
    }
    if(all_peds_[i]) all_peds_[i]->Reset();
    if(ped_rms_[i])ped_rms_[i]->Reset();
    if(ped_mean_[i])ped_mean_[i]->Reset();
    if(sub_rms_[i])ped_rms_[i]->Reset();
    if(sub_mean_[i])ped_mean_[i]->Reset();
    if(capid_mean_[i])capid_mean_[i]->Reset();
    if(capid_rms_[i])capid_rms_[i]->Reset();
    if(qie_mean_[i])qie_mean_[i]->Reset();
    if(qie_rms_[i])qie_rms_[i]->Reset();
    if(err_map_geo_[i])err_map_geo_[i]->Reset();
    if(err_map_elec_[i])err_map_elec_[i]->Reset();
  }

  return;
}

void HcalPedestalClient::htmlOutput(int runNo, string htmlDir, string htmlName){
  
  cout << "Preparing HcalPedestalClient html output ..." << endl;
  string client = "PedestalMonitor";
  generateBadChanList(htmlDir);
  htmlErrors(runNo,htmlDir,client,process_,dbe_,dqmReportMapErr_,dqmReportMapWarn_,dqmReportMapOther_);

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
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << runNo << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">Hcal Pedestals</span></h2> " << endl;

  htmlFile << "<h2>Events processed:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << ievt_ << "</span></h2>" << endl;

  htmlFile << "<hr>" << endl;
  htmlFile << "<table width=100% border=1><tr>" << endl;
  if(hasErrors())htmlFile << "<td bgcolor=red><a href=\"PedestalMonitorErrors.html\">Errors in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Errors</td>" << endl;
  if(hasWarnings()) htmlFile << "<td bgcolor=yellow><a href=\"PedestalMonitorWarnings.html\">Warnings in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Warnings</td>" << endl;
  if(hasOther()) htmlFile << "<td bgcolor=aqua><a href=\"PedestalMonitorMessages.html\">Messages in this task</a></td>" << endl;
  else htmlFile << "<td bgcolor=lime>No Messages</td>" << endl;
  htmlFile << "<td><a href=\"badPedestalList.html\">Pedestal Error List</a></td>" << endl;
  htmlFile << "</tr></table>" << endl;

  htmlFile << "<hr>" << endl;

  htmlFile << "<h2><strong>Hcal Pedestal Histograms</strong></h2>" << endl;
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
  histoHTML2(runNo,pedMapMeanD_[0],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,pedMapRMSD_[0],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,pedMapMeanD_[1],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,pedMapRMSD_[1],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,pedMapMeanD_[2],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,pedMapRMSD_[2],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;
  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,pedMapMeanD_[3],"iEta","iPhi", 92, htmlFile,htmlDir);
  histoHTML2(runNo,pedMapRMSD_[3],"iEta","iPhi", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,pedMapMean_E[0],"Crate","Slot", 92, htmlFile,htmlDir);
  histoHTML2(runNo,pedMapRMS_E[0],"Crate","Slot", 100, htmlFile,htmlDir);
  htmlFile << "</tr>" << endl;

  htmlFile << "<tr align=\"left\">" << endl;
  histoHTML2(runNo,pedMapMean_E[1],"Channel","Fiber Index", 92, htmlFile,htmlDir);
  histoHTML2(runNo,pedMapRMS_E[1],"Channel","Fiber Index", 100, htmlFile,htmlDir);
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
    histoHTML(runNo,ped_rms_[i],"Pedestal RMS (ADC)","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,ped_mean_[i],"Pedestal Mean (ADC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,sub_rms_[i],"Pedestal RMS (Nsigma)","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,sub_mean_[i],"Pedestal Mean (ADC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,capid_rms_[i],"Variance in CAPID RMS (ADC)","Events", 92, htmlFile,htmlDir);

    histoHTML(runNo,capid_mean_[i],"Variance in CAPID Mean (ADC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;

    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,qie_rms_[i],"Average QIE RMS (ADC)","Events", 92, htmlFile,htmlDir);
    histoHTML(runNo,qie_mean_[i],"Average QIE Mean (ADC)","Events", 100, htmlFile,htmlDir);
    htmlFile << "</tr>" << endl;
    
    htmlFile << "<tr align=\"left\">" << endl;
    histoHTML(runNo,all_peds_[i],"Pedestal Value (ADC)","Events",100, htmlFile,htmlDir);
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

  TNamed* tnd = (TNamed*)infile->Get("DQMData/Hcal/PedestalMonitor/Pedestal Task Event Number");
  if(tnd){
    string s =tnd->GetTitle();
    ievt_ = -1;
    sscanf((s.substr(2,s.length()-2)).c_str(), "%d", &ievt_);
  }

  char name[256];    
  
  for(int i=0; i<4; i++){
    sprintf(name,"DQMData/Hcal/PedestalMonitor/Ped Mean Depth %d",i+1);
    pedMapMeanD_[i] = (TH2F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/PedestalMonitor/Ped RMS Depth %d",i+1);
    pedMapRMSD_[i]  = (TH2F*)infile->Get(name);
  }
  
  sprintf(name,"DQMData/Hcal/PedestalMonitor/Ped Mean by Crate-Slot");
  pedMapMean_E[0] = (TH2F*)infile->Get(name);
  sprintf(name,"DQMData/Hcal/PedestalMonitor/Ped RMS by Crate-Slot");
  pedMapRMS_E[0] = (TH2F*)infile->Get(name);
  
  sprintf(name,"DQMData/Hcal/PedestalMonitor/Ped Mean by Fiber-Chan");
  pedMapMean_E[1] = (TH2F*)infile->Get(name);
  sprintf(name,"DQMData/Hcal/PedestalMonitor/Ped RMS by Fiber-Chan");
  pedMapRMS_E[1] = (TH2F*)infile->Get(name);
  
  for(int i=0; i<4; i++){
    if(!subDetsOn_[i]) continue;
    string type = "HB";
    if(i==1) type = "HE"; 
    else if(i==2) type = "HF";
    else if(i==3) type = "HO";
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s All Pedestal Values",type.c_str(),type.c_str());
    all_peds_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Pedestal RMS Values",type.c_str(),type.c_str());
    ped_rms_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Pedestal Mean Values",type.c_str(),type.c_str());
    ped_mean_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Normalized RMS Values",type.c_str(),type.c_str());
    sub_rms_[i] = (TH1F*)infile->Get(name);
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Subtracted Mean Values",type.c_str(),type.c_str());
    sub_mean_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s CapID RMS Variance",type.c_str(),type.c_str());
    capid_rms_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s CapID Mean Variance",type.c_str(),type.c_str());
    capid_mean_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s QIE RMS Values",type.c_str(),type.c_str());
    qie_rms_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s QIE Mean Values",type.c_str(),type.c_str());
    qie_mean_[i] = (TH1F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Pedestal Geo Error Map",type.c_str(),type.c_str());
    err_map_geo_[i] = (TH2F*)infile->Get(name);
    
    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Pedestal Elec Error Map",type.c_str(),type.c_str());
    err_map_elec_[i] = (TH2F*)infile->Get(name);
    
    bool capidOK = false;
    for(int ieta=-42; ieta<=42; ieta++){
      if(ieta==0) continue;
      for(int iphi=1; iphi<=73; iphi++){
	for(int depth=1; depth<=4; depth++){
	  if(!isValidGeom(i, ieta, iphi,depth)) continue;
	  
	  capidOK = true;
	  
	  HcalSubdetector subdet = HcalBarrel;
	  if(i==1) subdet = HcalEndcap;
	  else if(i==2) subdet = HcalForward;
	  else if(i==3) subdet = HcalOuter;
	  HcalDetId id(subdet,ieta,iphi,depth);
	  HcalElectronicsId eid;
	  
	  float capmeanS[4]; float caprmsS[4];	  
	  float capmeanP[4]; float caprmsP[4];	  
	  for(int capid=0; capid<4 && capidOK; capid++){
	    capmeanS[capid]=0; caprmsS[capid]=0; 
	    capmeanP[capid]=0; caprmsP[capid]=0; 
	    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Pedestal Value (ADC) ieta=%d iphi=%d depth=%d CAPID=%d", type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    TH1F* mePed = (TH1F*)infile->Get(name);
	    
	    sprintf(name,"DQMData/Hcal/PedestalMonitor/%s/%s Pedestal Value (Subtracted) ieta=%d iphi=%d depth=%d CAPID=%d", type.c_str(),type.c_str(),ieta,iphi,depth,capid);  
	    TH1F* meSub = (TH1F*)infile->Get(name);
	    
	    if(mePed!=NULL){
	      if(readoutMap_) eid = readoutMap_->lookup(id);	      
	      capmeanP[capid] = mePed->GetMean();
	      caprmsP[capid] = mePed->GetRMS();
	      capmeanS[capid] = meSub->GetMean();
	      caprmsS[capid] = meSub->GetRMS();
	      if(ped_rms_[i]) ped_rms_[i]->Fill(mePed->GetRMS());
	      if(ped_mean_[i]) ped_mean_[i]->Fill(mePed->GetMean());
	      
	      double width=1.0;
	      if(readoutMap_){
		const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
		if(pedw) width = pedw->getWidth(capid);
		if(width!=0) sub_rms_[i]->Fill(meSub->GetRMS()/width);
	      }
	      sub_mean_[i]->Fill(meSub->GetMean());
	      if(meSub->GetRMS()>pedrms_thresh_ || fabs(meSub->GetMean())>pedmean_thresh_) {
		if(err_map_geo_[i])err_map_geo_[i]->Fill(ieta,iphi);
		if(readoutMap_ && err_map_elec_[i]) err_map_elec_[i]->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	      }
	    }
	    else capidOK=false;
	    
	  }
	  float avgMean = 100; float avgRMS=100;
	  if(capidOK){
	    if(capid_mean_[i]) capid_mean_[i]->Fill(maxDiff(capmeanP[0],capmeanP[1],capmeanP[2],capmeanP[3]));
	    if(capid_rms_[i]) capid_rms_[i]->Fill(maxDiff(caprmsP[0],caprmsP[1],caprmsP[2],caprmsP[3]));
	    
	    if(maxDiff(capmeanP[0],capmeanP[1],capmeanP[2],capmeanP[3])>capmean_thresh_){
	      if(err_map_geo_[i]) err_map_geo_[i]->Fill(ieta,iphi);
	      if(readoutMap_ && err_map_elec_[i]) err_map_elec_[i]->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	    if(maxDiff(caprmsP[0],caprmsP[1],caprmsP[2],caprmsP[3])>caprms_thresh_){
	      if(err_map_geo_[i])err_map_geo_[i]->Fill(ieta,iphi);
	      if(readoutMap_ && err_map_elec_[i]) err_map_elec_[i]->Fill(eid.readoutVMECrateId(),eid.htrSlot());
	    }
	    
	    float avg = (capmeanP[0]+capmeanP[1]+capmeanP[2]+capmeanP[3])/4.0;
	    if(qie_mean_[i]) qie_mean_[i]->Fill(avg);
	    avg = (caprmsP[0]+caprmsP[1]+caprmsP[2]+caprmsP[3])/4.0;
	    if(qie_rms_[i]) qie_rms_[i]->Fill(avg);
	    
	    avgMean = (capmeanS[0]+capmeanS[1]+capmeanS[2]+capmeanS[3])/4.0;
	    avgRMS = (caprmsS[0]+caprmsS[1]+caprmsS[2]+caprmsS[3])/4.0;
	    if(plotPedRAW_){
	      avgMean = (capmeanP[0]+capmeanP[1]+capmeanP[2]+capmeanP[3])/4.0;
	      avgRMS = (caprmsP[0]+caprmsP[1]+caprmsP[2]+caprmsP[3])/4.0;
	    }
	  }
	  if(avgMean!=100 && depth>0){
	    if(pedMapMeanD_[depth-1]!=0) pedMapMeanD_[depth-1]->Fill(ieta,iphi,avgMean);
	    if(pedMapRMSD_[depth-1]!=0) pedMapRMSD_[depth-1]->Fill(ieta,iphi,avgRMS);
	    if(pedMapMean_E[0]!=0) pedMapMean_E[0]->Fill(eid.readoutVMECrateId(),eid.htrSlot(),avgMean);
	    if(pedMapMean_E[1]!=0) pedMapMean_E[1]->Fill(eid.fiberChanId(),eid.fiberIndex(),avgMean);
	    if(pedMapRMS_E[0]!=0) pedMapRMS_E[0]->Fill(eid.readoutVMECrateId(),eid.htrSlot(),avgRMS);
	    if(pedMapRMS_E[1]!=0) pedMapRMS_E[1]->Fill(eid.fiberChanId(),eid.fiberIndex(),avgRMS);
	  }
	}
      }      
    }
  }
  
  return;
}


void HcalPedestalClient::generateBadChanList(string htmlDir){
  if(!dbe_) return;
  
  if(doPerChanTests_){
    char name[256];
    ofstream outFile;
    outFile.open((htmlDir + "badPedestalList.html").c_str());
    outFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
    outFile << "<html>  " << endl;
    outFile << "<head>  " << endl;
    outFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
    outFile << " http-equiv=\"content-type\">  " << endl;
    outFile << "  <title>Pedestal Error List</title> " << endl;
    outFile << "</head>  " << endl;
    outFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
    outFile << "<body>  " << endl;
    outFile << "<br>  " << endl;
    outFile << "<hr>  " << endl;
    outFile << "<center><h2>Pedestal Errors</h2></center>" << endl;
    outFile << "<table WIDTH=100% border=\"10\" cellspacing=\"0\" cellpadding=\"10\"> " << endl;
    outFile << "<h2><strong><tr>" << endl;
    outFile << "<th> SubDet </th>" << endl;
    outFile << "<th> iEta </th>" << endl;
    outFile << "<th> iPhi </th>" << endl;
    outFile << "<th> Depth </th>" << endl;
    outFile << "<th> CapID </th>" << endl;
    outFile << "<th> Mean  (Ref)</th>" << endl;
    outFile << "<th> RMS  (Ref)</th>" << endl;
    outFile << " </tr></strong></h2><hr></br>" << endl;    

    for(int i=0; i<4; i++){
      if(!subDetsOn_[i]) continue;
      string type = "HB";
      if(i==1) type = "HE"; 
      else if(i==2) type = "HF";
      else if(i==3) type = "HO";

      for(int ieta=-42; ieta<=42; ieta++){
	if(ieta==0) continue;
	for(int iphi=1; iphi<=73; iphi++){
	  for(int depth=1; depth<=4; depth++){
	    if(!isValidGeom(i, ieta, iphi,depth)) continue;
	    for(int capid=0; capid<4; capid++){
	      sprintf(name,"%s Pedestal ieta=%d iphi=%d depth=%d CAPID=%d: Sigma",type.c_str(),ieta,iphi,depth,capid);  
	      map<string, string>::iterator errTest=dqmQtests_.find(name);
	      if( errTest != dqmQtests_.end()){ 
		string testName = errTest->first;
		string meName = errTest->second;
		MonitorElement* me = dbe_->get(meName);
		if(me){
		  HcalSubdetector subdet = HcalBarrel;
		  if(i==1) subdet = HcalEndcap;
		  else if(i==2) subdet = HcalForward;
		  else if(i==3) subdet = HcalOuter;
		  const HcalDetId id(subdet,ieta,iphi,depth);
		  const HcalPedestalWidth* pedw = (*conditions_).getPedestalWidth(id);
		  const HcalPedestal* pedm = (*conditions_).getPedestal(id);

		  if (me->hasError()){
		    outFile << "<h3><font color=red><tr>" << endl;
		    outFile << "<td> "<< type.c_str() <<" </td>" << endl;
		    outFile << "<td> "<< ieta <<" </td>" << endl;
		    outFile << "<td> "<< iphi <<" </td>" << endl;
		    outFile << "<td> "<< depth<<" </td>" << endl;
		    outFile << "<td> "<< capid<<" </td>" << endl;
		    sprintf(name,"%.2f  ( %.2f )",me->getMean(), pedm->getValue(capid));
		    outFile << "<td> "<< name << " </td>" << endl;
		    sprintf(name,"%.2f  ( %.2f )",me->getRMS(), pedw->getWidth(capid));
		    outFile << "<td> "<< name << " </td>" << endl;
		    outFile << "</tr></h3></font>" << endl;
		  }
		  if (me->hasWarning()){
		    outFile << "<h3><font color=blue><tr>" << endl;
		    outFile << "<td> "<< type.c_str() <<" </td>" << endl;
		    outFile << "<td> "<< ieta <<" </td>" << endl;
		    outFile << "<td> "<< iphi <<" </td>" << endl;
		    outFile << "<td> "<< depth<<" </td>" << endl;
		    outFile << "<td> "<< capid<<" </td>" << endl;
		    sprintf(name,"%.2f  ( %.2f )",me->getMean(), pedm->getValue(capid));
		    outFile << "<td> "<< name << " </td>" << endl;
		    sprintf(name,"%.2f  ( %.2f )",me->getRMS(), pedw->getWidth(capid));
		    outFile << "<td> "<< name << " </td>" << endl;
		    outFile << "</tr></h3></font>" << endl;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    outFile << "</table>" << endl;
    outFile << "<hr>" << endl;
    outFile.close();
  }
   
  return;
}
