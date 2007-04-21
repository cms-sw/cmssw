#include "DQM/HcalMonitorTasks/interface/HcalLEDMonitor.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"

HcalLEDMonitor::HcalLEDMonitor() {
  doPerChannel_ = false;
  sigS0_=0;
  sigS1_=9;

}

HcalLEDMonitor::~HcalLEDMonitor() {}

void HcalLEDMonitor::clearME(){
     if ( m_dbe ) {
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HF");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HO");
    m_dbe->removeContents();
  }
}

void HcalLEDMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( ps.getUntrackedParameter<bool>("LEDPerChannel", false) ) {
    doPerChannel_ = true;
  }
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);

  sigS0_ = ps.getUntrackedParameter<int>("FirstSignalBin", 0);
  sigS1_ = ps.getUntrackedParameter<int>("LastSignalBin", 9);

  if(sigS0_<0){
    printf("HcalLEDMonitor::setup, illegal range for first sample: %d\n",sigS0_);
    sigS0_=0;
  }
  if(sigS1_>9){
    printf("HcalLEDMonitor::setup, illegal range for last sample: %d\n",sigS1_);
    sigS1_=9;
  }

  if(sigS0_>sigS1_){ 
    printf("HcalLEDMonitor::setup, illegal range for first/last sample: %d/%d\n",sigS0_,sigS1_);
    sigS0_=0; sigS1_=9;
  }

  ievt_=0;

  if ( m_dbe ) {

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor");
    meEVT_ = m_dbe->bookInt("LED Task Event Number");    
    meEVT_->Fill(ievt_);

    MEAN_MAP_TIME_L1= m_dbe->book2D("LED Mean Time Depth 1","LED Mean Time Depth 1",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_TIME_L1= m_dbe->book2D("LED RMS Time Depth 1","LED RMS Time Depth 1",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_TIME_L2= m_dbe->book2D("LED Mean Time Depth 2","LED Mean Time Depth 2",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_TIME_L2= m_dbe->book2D("LED RMS Time Depth 2","LED RMS Time Depth 2",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_TIME_L3= m_dbe->book2D("LED Mean Time Depth 3","LED Mean Time Depth 3",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_TIME_L3= m_dbe->book2D("LED RMS Time Depth 3","LED RMS Time Depth 3",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_TIME_L4= m_dbe->book2D("LED Mean Time Depth 4","LED Mean Time Depth 4",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_TIME_L4= m_dbe->book2D("LED RMS Time Depth 4","LED RMS Time Depth 4",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);


    MEAN_MAP_SHAPE_L1= m_dbe->book2D("LED Mean Shape Depth 1","LED Mean Shape Depth 1",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_SHAPE_L1= m_dbe->book2D("LED RMS Shape Depth 1","LED RMS Shape Depth 1",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_SHAPE_L2= m_dbe->book2D("LED Mean Shape Depth 2","LED Mean Shape Depth 2",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_SHAPE_L2= m_dbe->book2D("LED RMS Shape Depth 2","LED RMS Shape Depth 2",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_SHAPE_L3= m_dbe->book2D("LED Mean Shape Depth 3","LED Mean Shape Depth 3",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_SHAPE_L3= m_dbe->book2D("LED RMS Shape Depth 3","LED RMS Shape Depth 3",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_SHAPE_L4= m_dbe->book2D("LED Mean Shape Depth 4","LED Mean Shape Depth 4",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_SHAPE_L4= m_dbe->book2D("LED RMS Shape Depth 4","LED RMS Shape Depth 4",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);

    MEAN_MAP_ENERGY_L1= m_dbe->book2D("LED Mean Energy Depth 1","LED Mean Energy Depth 1",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_ENERGY_L1= m_dbe->book2D("LED RMS Energy Depth 1","LED RMS Energy Depth 1",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_ENERGY_L2= m_dbe->book2D("LED Mean Energy Depth 2","LED Mean Energy Depth 2",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_ENERGY_L2= m_dbe->book2D("LED RMS Energy Depth 2","LED RMS Energy Depth 2",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_ENERGY_L3= m_dbe->book2D("LED Mean Energy Depth 3","LED Mean Energy Depth 3",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_ENERGY_L3= m_dbe->book2D("LED RMS Energy Depth 3","LED RMS Energy Depth 3",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_ENERGY_L4= m_dbe->book2D("LED Mean Energy Depth 4","LED Mean Energy Depth 4",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_ENERGY_L4= m_dbe->book2D("LED RMS Energy Depth 4","LED RMS Energy Depth 4",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HB");
    hbHists.shapePED =  m_dbe->book1D("HB Ped Subtracted Pulse Shape","HB Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hbHists.shapeALL =  m_dbe->book1D("HB Average Pulse Shape","HB Average Pulse Shape",10,-0.5,9.5);
    hbHists.rms_shape =  m_dbe->book1D("HB LED Shape RMS Values","HB LED Shape RMS Values",100,0,5);
    hbHists.mean_shape =  m_dbe->book1D("HB LED Shape Mean Values","HB LED Shape Mean Values",100,-0.5,9.5);
    hbHists.timeALL =  m_dbe->book1D("HB Average Pulse Time","HB Average Pulse Time",200,-1,10);
    hbHists.rms_time =  m_dbe->book1D("HB LED Time RMS Values","HB LED Time RMS Values",100,0,5);
    hbHists.mean_time =  m_dbe->book1D("HB LED Time Mean Values","HB LED Time Mean Values",100,-1,10);
    hbHists.energyALL =  m_dbe->book1D("HB Average Pulse Energy","HB Average Pulse Energy",500,0,5000);
    hbHists.rms_energy =  m_dbe->book1D("HB LED Energy RMS Values","HB LED Energy RMS Values",100,0,500);
    hbHists.mean_energy =  m_dbe->book1D("HB LED Energy Mean Values","HB LED Energy Mean Values",100,0,1000);

    hbHists.err_map_geo =  m_dbe->book2D("HB LED Geo Error Map","HB LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.err_map_elec =  m_dbe->book2D("HB LED Elec Error Map","HB LED Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HE");
    heHists.shapePED =  m_dbe->book1D("HE Ped Subtracted Pulse Shape","HE Ped Subtracted Pulse Shape",10,-0.5,9.5);
    heHists.shapeALL =  m_dbe->book1D("HE Average Pulse Shape","HE Average Pulse Shape",10,-0.5,9.5);
    heHists.timeALL =  m_dbe->book1D("HE Average Pulse Time","HE Average Pulse Time",200,-1,10);
    heHists.rms_shape =  m_dbe->book1D("HE LED Shape RMS Values","HE LED Shape RMS Values",100,0,5);
    heHists.mean_shape =  m_dbe->book1D("HE LED Shape Mean Values","HE LED Shape Mean Values",100,-0.5,9.5);
    heHists.rms_time =  m_dbe->book1D("HE LED Time RMS Values","HE LED Time RMS Values",100,0,5);
    heHists.mean_time =  m_dbe->book1D("HE LED Time Mean Values","HE LED Time Mean Values",100,-1,10);
    heHists.energyALL =  m_dbe->book1D("HE Average Pulse Energy","HE Average Pulse Energy",500,0,5000);
    heHists.rms_energy =  m_dbe->book1D("HE LED Energy RMS Values","HE LED Energy RMS Values",100,0,500);
    heHists.mean_energy =  m_dbe->book1D("HE LED Energy Mean Values","HE LED Energy Mean Values",100,0,1000);
heHists.err_map_geo =  m_dbe->book2D("HE LED Geo Error Map","HE LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.err_map_elec =  m_dbe->book2D("HE LED Elec Error Map","HE LED Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HF");
    hfHists.shapePED =  m_dbe->book1D("HF Ped Subtracted Pulse Shape","HF Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hfHists.shapeALL =  m_dbe->book1D("HF Average Pulse Shape","HF Average Pulse Shape",10,-0.5,9.5);
    hfHists.timeALL =  m_dbe->book1D("HF Average Pulse Time","HF Average Pulse Time",200,-1,10);
    hfHists.rms_shape =  m_dbe->book1D("HF LED Shape RMS Values","HF LED Shape RMS Values",100,0,5);
    hfHists.mean_shape =  m_dbe->book1D("HF LED Shape Mean Values","HF LED Shape Mean Values",100,-0.5,9.5);
    hfHists.rms_time =  m_dbe->book1D("HF LED Time RMS Values","HF LED Time RMS Values",100,0,5);
    hfHists.mean_time =  m_dbe->book1D("HF LED Time Mean Values","HF LED Time Mean Values",100,-1,10);
    hfHists.energyALL =  m_dbe->book1D("HF Average Pulse Energy","HF Average Pulse Energy",500,0,5000);
    hfHists.rms_energy =  m_dbe->book1D("HF LED Energy RMS Values","HF LED Energy RMS Values",100,0,500);
    hfHists.mean_energy =  m_dbe->book1D("HF LED Energy Mean Values","HF LED Energy Mean Values",100,0,1000);
hfHists.err_map_geo =  m_dbe->book2D("HF LED Geo Error Map","HF LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.err_map_elec =  m_dbe->book2D("HF LED Elec Error Map","HF LED Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HO");
    hoHists.shapePED =  m_dbe->book1D("HO Ped Subtracted Pulse Shape","HO Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hoHists.shapeALL =  m_dbe->book1D("HO Average Pulse Shape","HO Average Pulse Shape",10,-0.5,9.5);
    hoHists.timeALL =  m_dbe->book1D("HO Average Pulse Time","HO Average Pulse Time",200,-1,10);
    hoHists.rms_shape =  m_dbe->book1D("HO LED Shape RMS Values","HO LED Shape RMS Values",100,0,5);
    hoHists.mean_shape =  m_dbe->book1D("HO LED Shape Mean Values","HO LED Shape Mean Values",100,-0.5,9.5);
    hoHists.rms_time =  m_dbe->book1D("HO LED Time RMS Values","HO LED Time RMS Values",100,0,5);
    hoHists.mean_time =  m_dbe->book1D("HO LED Time Mean Values","HO LED Time Mean Values",100,-1,10);
    hoHists.energyALL =  m_dbe->book1D("HO Average Pulse Energy","HO Average Pulse Energy",500,0,5000);
    hoHists.rms_energy =  m_dbe->book1D("HO LED Energy RMS Values","HO LED Energy RMS Values",100,0,500);
    hoHists.mean_energy =  m_dbe->book1D("HO LED Energy Mean Values","HO LED Energy Mean Values",100,0,1000);
    hoHists.err_map_geo =  m_dbe->book2D("HO LED Geo Error Map","HO LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.err_map_elec =  m_dbe->book2D("HO LED Elec Error Map","HO LED Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);
  }

  return;
}

void HcalLEDMonitor::processEvent(const HBHEDigiCollection& hbhe,
				  const HODigiCollection& ho,
				  const HFDigiCollection& hf,
				  const HcalDbService& cond){
  

  ievt_++;
  meEVT_->Fill(ievt_);

  if(!m_dbe) { printf("HcalLEDMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }
  float vals[10];

  try{
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      
      cond.makeHcalCalibration(digi.id(), &calibs_);
      float t =0; float b=0;
      for(int i=sigS0_; i<=sigS1_; i++){	  
	t += i*(digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	b += digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
      }
      if(digi.id().subdet()==HcalBarrel){
	hbHists.energyALL->Fill(b);
	if(b!=0) hbHists.timeALL->Fill(t/b);
	for (int i=0; i<digi.size(); i++) {
	  hbHists.shapeALL->Fill(i,digi.sample(i).adc());
	  hbHists.shapePED->Fill(i,digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	  vals[i] = digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
	}
	if(doPerChannel_) perChanHists(0,digi.id(),vals,hbHists.shape, hbHists.time, hbHists.energy);
      }
      else if(digi.id().subdet()==HcalEndcap){
	heHists.energyALL->Fill(b);
	if(b!=0) heHists.timeALL->Fill(t/b);
	for (int i=0; i<digi.size(); i++) {
	  heHists.shapeALL->Fill(i,digi.sample(i).adc());
	  heHists.shapePED->Fill(i,digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	  vals[i] = digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
	}
	if(doPerChannel_) perChanHists(1,digi.id(),vals,heHists.shape, heHists.time, heHists.energy);
      }

    }
  } catch (...) {
    printf("HcalLEDMonitor::processEvent  No HBHE Digis.\n");
  }
  
  try{
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);	
      cond.makeHcalCalibration(digi.id(), &calibs_);
      float t =0; float b=0;
      for(int i=sigS0_; i<=sigS1_; i++){	  
	t += i*(digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	b += digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
      }
      hoHists.energyALL->Fill(b);
      if(b!=0) hoHists.timeALL->Fill(t/b);
      for (int i=0; i<digi.size(); i++) {
	hoHists.shapeALL->Fill(i,digi.sample(i).adc());
	hoHists.shapePED->Fill(i,digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	vals[i] = digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
      }
      if(doPerChannel_) perChanHists(2,digi.id(),vals,hoHists.shape, hoHists.time, hoHists.energy);
    }        
  } catch (...) {
    cout << "HcalLEDMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);
      cond.makeHcalCalibration(digi.id(), &calibs_);
      float t =0; float b=0;
      for(int i=sigS0_; i<=sigS1_; i++){	  
	t += i*(digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	b += digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
      }
      hfHists.energyALL->Fill(b);
      if(b!=0) hfHists.timeALL->Fill(t/b);
      for (int i=0; i<digi.size(); i++) {
	hfHists.shapeALL->Fill(i,digi.sample(i).adc());
	hfHists.shapePED->Fill(i,digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid()));
	vals[i] = digi.sample(i).adc()-calibs_.pedestal(digi.sample(i).capid());
      }
      if(doPerChannel_) perChanHists(3,digi.id(),vals,hfHists.shape, hfHists.time, hfHists.energy);
    }
  } catch (...) {
    cout << "HcalLEDMonitor::processEvent  No HF Digis." << endl;
  }

  return;

}

void HcalLEDMonitor::done(){
  return;
}

void HcalLEDMonitor::perChanHists(int id, const HcalDetId detid, float* vals, 
				  map<HcalDetId, MonitorElement*> &tShape, 
				  map<HcalDetId, MonitorElement*> &tTime,
				  map<HcalDetId, MonitorElement*> &tEnergy){
  
  MonitorElement* _me;
  if(m_dbe==NULL) return;

  _meo=tShape.begin();
  string type = "HB";
  m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HB");
  if(id==1){
    type = "HE";
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HE");
  }
  else if(id==2){
    type = "HO";
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HO");
  }
  else if(id==3){
    type = "HF";
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HF");
  }
  
  _meo = tShape.find(detid);
  if (_meo!=tShape.end()){
    _me= _meo->second;
    if(_me==NULL) printf("HcalLEDAnalysis::perChanHists  This histo is NULL!!??\n");
    else{
      
      float top=0; float bottom = 0;
      for(int i=sigS0_; i<=sigS1_; i++){
	top += i*vals[i];
	bottom += vals[i];
	_me->Fill(i,vals[i]); 
      }
      _me = tTime[detid];      
      if(bottom!=0) _me->Fill(top/bottom); 
      _me = tEnergy[detid];  
      _me->Fill(bottom); 
    }
  }
  else{
    char name[1024];
    sprintf(name,"%s LED Shape ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert1;
    insert1 =  m_dbe->book1D(name,name,10,-0.5,9.5);
    float top=0; float bottom = 0;
    for(int i=sigS0_; i<=sigS1_; i++){
      top += i*vals[i];
      bottom += vals[i];
      insert1->Fill(i,vals[i]); 
    }
    tShape[detid] = insert1;
    
    sprintf(name,"%s LED Time ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert2 =  m_dbe->book1D(name,name,100,0,10);
    if(bottom!=0) insert2->Fill(top/bottom); 
    tTime[detid] = insert2;	

    sprintf(name,"%s LED Energy ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert3 =  m_dbe->book1D(name,name,250,0,5000);
    if(bottom!=0) insert3->Fill(bottom); 
    tEnergy[detid] = insert3;	
    
  } 

  return;

}  
