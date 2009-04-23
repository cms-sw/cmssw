#include "DQM/HcalMonitorTasks/interface/HcalLEDMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalLEDMonitor::HcalLEDMonitor() {
  doPerChannel_ = false;
  sigS0_=0;
  sigS1_=9;
}

HcalLEDMonitor::~HcalLEDMonitor() {}

void HcalLEDMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  baseFolder_ = rootFolder_+"LEDMonitor";

  if ( ps.getUntrackedParameter<bool>("LEDPerChannel", false) ) {
    doPerChannel_ = true;
  }
  if (fVerbosity>0)
    cout << "LED Monitor eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  if (fVerbosity>0)
    cout << "LED Monitor phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  sigS0_ = ps.getUntrackedParameter<int>("FirstSignalBin", 0);
  sigS1_ = ps.getUntrackedParameter<int>("LastSignalBin", 9);
  
  adcThresh_ = ps.getUntrackedParameter<double>("LED_ADC_Thresh", 0);
  if (fVerbosity>0) cout << "LED Monitor threshold set to " << adcThresh_ << endl;
  if (fVerbosity>0) cout << "LED Monitor signal window set to " << sigS0_ <<"-"<< sigS1_ << endl;  

  if(sigS0_<0){
    if (fVerbosity>0) cout<<"HcalLEDMonitor::setup, illegal range for first sample: "<<sigS0_<<endl;
    sigS0_=0;
  }
  if(sigS1_>9){
    if (fVerbosity>0) cout <<"HcalLEDMonitor::setup, illegal range for last sample: "<<sigS1_<<endl;
    sigS1_=9;
  }

  if(sigS0_>sigS1_){ 
    if (fVerbosity>0) cout <<"HcalLEDMonitor::setup, illegal range for first/last sample: "<<sigS0_<<"/"<<sigS1_<<endl;
    sigS0_=0; sigS1_=9;
  }

  ievt_=0;

  if ( m_dbe ) {

    m_dbe->setCurrentFolder(baseFolder_);
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

    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    hbHists.shapePED =  m_dbe->book1D("HB Ped Subtracted Pulse Shape","HB Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hbHists.shapeALL =  m_dbe->book1D("HB Average Pulse Shape","HB Average Pulse Shape",10,-0.5,9.5);
    hbHists.rms_shape =  m_dbe->book1D("HB LED Shape RMS Values","HB LED Shape RMS Values",100,0,5);
    hbHists.mean_shape =  m_dbe->book1D("HB LED Shape Mean Values","HB LED Shape Mean Values",100,-0.5,9.5);

    hbHists.timeALL =  m_dbe->book1D("HB Average Pulse Time","HB Average Pulse Time",200,-0.5,9.5);
    hbHists.rms_time =  m_dbe->book1D("HB LED Time RMS Values","HB LED Time RMS Values",100,0,5);
    hbHists.mean_time =  m_dbe->book1D("HB LED Time Mean Values","HB LED Time Mean Values",100,-0.5,9.5);

    hbHists.energyALL =  m_dbe->book1D("HB Average Pulse Energy","HB Average Pulse Energy",500,0,5000);
    hbHists.rms_energy =  m_dbe->book1D("HB LED Energy RMS Values","HB LED Energy RMS Values",200,0,1000);
    hbHists.mean_energy =  m_dbe->book1D("HB LED Energy Mean Values","HB LED Energy Mean Values",100,0,1000);

    hbHists.err_map_geo =  m_dbe->book2D("HB LED Geo Error Map","HB LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.err_map_elec =  m_dbe->book2D("HB LED Elec Error Map","HB LED Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);

    m_dbe->setCurrentFolder(baseFolder_+"/HE");
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

    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    hfHists.shapePED =  m_dbe->book1D("HF Ped Subtracted Pulse Shape","HF Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hfHists.shapeALL =  m_dbe->book1D("HF Average Pulse Shape","HF Average Pulse Shape",10,-0.5,9.5);
    hfHists.timeALL =  m_dbe->book1D("HF Average Pulse Time","HF Average Pulse Time",200,-0.5,9.5);
    hfHists.rms_shape =  m_dbe->book1D("HF LED Shape RMS Values","HF LED Shape RMS Values",100,0,5);

    hfHists.mean_shape =  m_dbe->book1D("HF LED Shape Mean Values","HF LED Shape Mean Values",100,-0.5,9.5);
    hfHists.rms_time =  m_dbe->book1D("HF LED Time RMS Values","HF LED Time RMS Values",100,0,5);
    hfHists.mean_time =  m_dbe->book1D("HF LED Time Mean Values","HF LED Time Mean Values",100,-0.5,9.5);

    hfHists.energyALL =  m_dbe->book1D("HF Average Pulse Energy","HF Average Pulse Energy",1000,-50,5000);
    hfHists.rms_energy =  m_dbe->book1D("HF LED Energy RMS Values","HF LED Energy RMS Values",100,0,300);
    hfHists.mean_energy =  m_dbe->book1D("HF LED Energy Mean Values","HF LED Energy Mean Values",100,0,500);

    hfHists.err_map_geo =  m_dbe->book2D("HF LED Geo Error Map","HF LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.err_map_elec =  m_dbe->book2D("HF LED Elec Error Map","HF LED Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);


    //HFlumi plots
    HFlumi_ETsum_perwedge =  m_dbe->book1D("HF lumi ET-sum per wedge","HF lumi ET-sum per wedge",36,1,37);

    HFlumi_Occupancy_above_thr_r1 =  m_dbe->book1D("HF lumi Occupancy above threshold ring1","HF lumi Occupancy above threshold ring1",36,1,37);
    HFlumi_Occupancy_between_thrs_r1 = m_dbe->book1D("HF lumi Occupancy between thresholds ring1","HF lumi Occupancy between thresholds ring1",36,1,37);
    HFlumi_Occupancy_below_thr_r1 = m_dbe->book1D("HF lumi Occupancy below threshold ring1","HF lumi Occupancy below threshold ring1",36,1,37);
    HFlumi_Occupancy_above_thr_r2 = m_dbe->book1D("HF lumi Occupancy above threshold ring2","HF lumi Occupancy above threshold ring2",36,1,37);
    HFlumi_Occupancy_between_thrs_r2 = m_dbe->book1D("HF lumi Occupancy between thresholds ring2","HF lumi Occupancy between thresholds ring2",36,1,37);
    HFlumi_Occupancy_below_thr_r2 = m_dbe->book1D("HF lumi Occupancy below threshold ring2","HF lumi Occupancy below threshold ring2",36,1,37);

    m_dbe->setCurrentFolder(baseFolder_+"/HO");
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

void HcalLEDMonitor::createFEDmap(unsigned int fed){
  _fedIter = MEAN_MAP_SHAPE_DCC.find(fed);
  
  if(_fedIter==MEAN_MAP_SHAPE_DCC.end()){
    m_dbe->setCurrentFolder(baseFolder_);
    char name[256];

    sprintf(name,"DCC %d Mean Shape Map",fed);
    MonitorElement* mean_shape = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    sprintf(name,"DCC %d RMS Shape Map",fed);
    MonitorElement* rms_shape = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);

    MEAN_MAP_SHAPE_DCC[fed] = mean_shape;
    RMS_MAP_SHAPE_DCC[fed] = rms_shape;

    sprintf(name,"DCC %d Mean Time Map",fed);
    MonitorElement* mean_time = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    sprintf(name,"DCC %d RMS Time Map",fed);
    MonitorElement* rms_time = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    MEAN_MAP_TIME_DCC[fed] = mean_time;
    RMS_MAP_TIME_DCC[fed] = rms_time;
    
    sprintf(name,"DCC %d Mean Energy Map",fed);
    MonitorElement* mean_energy = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    sprintf(name,"DCC %d RMS Energy Map",fed);
    MonitorElement* rms_energy = m_dbe->book2D(name,name,24,0.5,24.5,15,0.5,15.5);
    MEAN_MAP_ENERGY_DCC[fed] = mean_energy;
    RMS_MAP_ENERGY_DCC[fed] = rms_energy;
  }   
}

void HcalLEDMonitor::reset(){
  
  MonitorElement* unpackedFEDS = m_dbe->get("Hcal/FEDs Unpacked");
  if(unpackedFEDS){
    for(int b=1; b<=unpackedFEDS->getNbinsX(); b++){
      if(unpackedFEDS->getBinContent(b)>0){
	createFEDmap(700+(b-1));  
      }
    }
  }
}

void HcalLEDMonitor::processEvent(const HBHEDigiCollection& hbhe,
				  const HODigiCollection& ho,
				  const HFDigiCollection& hf,
				  const HcalDbService& cond){

  ievt_++;
  meEVT_->Fill(ievt_);

  if(!m_dbe) { 
    if(fVerbosity) cout <<"HcalLEDMonitor::processEvent   DQMStore not instantiated!!!"<<endl;  
    return; }
  float vals[10];


  for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++)
    {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      
      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
      float en=0;
      float ts =0; float bs=0;
      int maxi=0; float maxa=0;
      for(int i=sigS0_; i<=sigS1_; i++){
	if(digi.sample(i).adc()>maxa){maxa=digi.sample(i).adc(); maxi=i;}
      }
      for(int i=sigS0_; i<=sigS1_; i++){	  
	float tmp1 =0;   
        int j1=digi.sample(i).adc();
        tmp1 = (LedMonAdc2fc[j1]+0.5);   	  
	en += tmp1-calibs_.pedestal(digi.sample(i).capid());
	if(i>=(maxi-1) && i<=maxi+1){
	  ts += i*(tmp1-calibs_.pedestal(digi.sample(i).capid()));
	  bs += tmp1-calibs_.pedestal(digi.sample(i).capid());
	}
      }
      if(en<adcThresh_) continue;
      if(digi.id().subdet()==HcalBarrel){
	hbHists.energyALL->Fill(en);
	if(bs!=0) hbHists.timeALL->Fill(ts/bs);
	for (int i=0; i<digi.size(); i++) {
	  float tmp =0;
	  int j=digi.sample(i).adc();
	  tmp = (LedMonAdc2fc[j]+0.5);
	  hbHists.shapeALL->Fill(i,tmp);
	  hbHists.shapePED->Fill(i,tmp-calibs_.pedestal(digi.sample(i).capid()));
	  vals[i] = tmp-calibs_.pedestal(digi.sample(i).capid());
	}
	if(doPerChannel_) perChanHists(0,digi.id(),vals,hbHists.shape, hbHists.time, hbHists.energy, baseFolder_);
      }
      else if(digi.id().subdet()==HcalEndcap){
	heHists.energyALL->Fill(en);
	if(bs!=0) heHists.timeALL->Fill(ts/bs);
	for (int i=0; i<digi.size(); i++) {
	  float tmp =0;
	  int j=digi.sample(i).adc();
	  tmp = (LedMonAdc2fc[j]+0.5);
	  heHists.shapeALL->Fill(i,tmp);
	  heHists.shapePED->Fill(i,tmp-calibs_.pedestal(digi.sample(i).capid()));
	  vals[i] = tmp-calibs_.pedestal(digi.sample(i).capid());
	}
	if(doPerChannel_) perChanHists(1,digi.id(),vals,heHists.shape, heHists.time, heHists.energy, baseFolder_);
      }
    } // for (HBHEDigiCollection...)

  
  for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++)
    {
      const HODataFrame digi = (const HODataFrame)(*j);	
      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
      float en=0;
      float ts =0; float bs=0;
      int maxi=0; float maxa=0;
      for(int i=sigS0_; i<=sigS1_; i++){
	if(digi.sample(i).adc()>maxa){maxa=digi.sample(i).adc(); maxi=i;}
      }
      for(int i=sigS0_; i<=sigS1_; i++){	  
	float tmp1 =0;   
        int j1=digi.sample(i).adc();
        tmp1 = (LedMonAdc2fc[j1]+0.5);   	  
	en += tmp1-calibs_.pedestal(digi.sample(i).capid());
	if(i>=(maxi-1) && i<=maxi+1){
	  ts += i*(tmp1-calibs_.pedestal(digi.sample(i).capid()));
	  bs += tmp1-calibs_.pedestal(digi.sample(i).capid());
	}
      }
      if(en<adcThresh_) continue;
      hoHists.energyALL->Fill(en);
      if(bs!=0) hoHists.timeALL->Fill(ts/bs);
      for (int i=0; i<digi.size(); i++) {
	float tmp =0;
        int j=digi.sample(i).adc();
        tmp = (LedMonAdc2fc[j]+0.5);
        hoHists.shapeALL->Fill(i,tmp);
        hoHists.shapePED->Fill(i,tmp-calibs_.pedestal(digi.sample(i).capid()));
	vals[i] = tmp-calibs_.pedestal(digi.sample(i).capid());
      }
      if(doPerChannel_) perChanHists(2,digi.id(),vals,hoHists.shape, hoHists.time, hoHists.energy, baseFolder_);
    } // for (HODigiCollection...)       
  
  for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++)
    {
      const HFDataFrame digi = (const HFDataFrame)(*j);
      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private. 
      float en=0;
      float ts =0; float bs=0;
      int maxi=0; float maxa=0;
      for(int i=sigS0_; i<=sigS1_; i++){
	if(digi.sample(i).adc()>maxa){maxa=digi.sample(i).adc(); maxi=i;}
      }
      for(int i=sigS0_; i<=sigS1_; i++){	  
	float tmp1 =0;   
        int j1=digi.sample(i).adc();
        tmp1 = (LedMonAdc2fc[j1]+0.5);   	  
	en += tmp1-calibs_.pedestal(digi.sample(i).capid());
	if(i>=(maxi-1) && i<=maxi+1){
	  ts += i*(tmp1-calibs_.pedestal(digi.sample(i).capid()));
	  bs += tmp1-calibs_.pedestal(digi.sample(i).capid());
	}
      }

      //---HFlumiplots
      int theTStobeused = 6;
      // will have masking later:
      int mask=1; 
      if(mask!=1) continue;
      //if we want to sum the 10 TS instead of just taking one:
      for (int i=0; i<digi.size(); i++) {
	if (i==theTStobeused) {
	  float tmpET =0;
	  int jadc=digi.sample(i).adc();
	  //NOW LUT used in HLX are only identy LUTs, so Et filled
	  //with unlinearised adc, ie tmpET = jadc
	  //	  tmpET = (adc2fc[jadc]+0.5);
	  tmpET = jadc;

	  //-find which wedge we are in
	  //  ETsum and Occupancy will be summed for both L and S
	  if(digi.id().ieta()>28){
	    if((digi.id().iphi()==1)||(digi.id().iphi()==71)){
	      HFlumi_ETsum_perwedge->Fill(1,tmpET);
              if((digi.id().ieta()==33)||(digi.id().ieta()==34)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(1,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(1,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(1,1);
	      }
	      else if((digi.id().ieta()==35)||(digi.id().ieta()==36)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(1,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(1,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(1,1);
	      }
	    }
	    else {
	      for (int iwedge=2; iwedge<19; iwedge++) {
		int itmp=4*(iwedge-1);
		if( (digi.id().iphi()==(itmp+1)) || (digi.id().iphi()==(itmp-1))) {
                  HFlumi_ETsum_perwedge->Fill(iwedge,tmpET);
		  if((digi.id().ieta()==33)||(digi.id().ieta()==34)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(iwedge,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(iwedge,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(iwedge,1);
		  }
		  else if((digi.id().ieta()==35)||(digi.id().ieta()==36)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(iwedge,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(iwedge,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(iwedge,1);
		  }
                  iwedge=99;
		}
	      }
	    }
	  }  //--endif ieta in HF+
	  else if(digi.id().ieta()<-28){
	    if((digi.id().iphi()==1)||(digi.id().iphi()==71)){
	      HFlumi_ETsum_perwedge->Fill(19,tmpET);
              if((digi.id().ieta()==-33)||(digi.id().ieta()==-34)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(19,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(19,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(19,1);
	      }
	      else if((digi.id().ieta()==-35)||(digi.id().ieta()==-36)) {
		if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(19,1);
		if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(19,1);
		if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(19,1);
	      }
	    }
	    else {
	      for (int iw=2; iw<19; iw++) {
		int itemp=4*(iw-1);
		if( (digi.id().iphi()==(itemp+1)) || (digi.id().iphi()==(itemp-1))) {
                  HFlumi_ETsum_perwedge->Fill(iw+18,tmpET);
		  if((digi.id().ieta()==-33)||(digi.id().ieta()==-34)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r1->Fill(iw+18,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r1->Fill(iw+18,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r1->Fill(iw+18,1);
		  }
		  else if((digi.id().ieta()==-35)||(digi.id().ieta()==-36)) {
		    if(jadc>100) HFlumi_Occupancy_above_thr_r2->Fill(iw+18,1);
		    if((jadc>=10)&&(jadc<=100)) HFlumi_Occupancy_between_thrs_r2->Fill(iw+18,1);
		    if(jadc<10) HFlumi_Occupancy_below_thr_r2->Fill(iw+18,1);
		  }
                  iw=99;
		}
	      }
	    }
	  }//---endif ieta inHF-
	}//---endif TS=nr6
      } //------end loop over TS for lumi
 
      if(en<adcThresh_) continue;
      hfHists.energyALL->Fill(en);
      if(bs!=0) hfHists.timeALL->Fill(ts/bs);
      for (int i=0; i<digi.size(); i++) {
        float tmp =0;
        int j=digi.sample(i).adc();
        tmp = (LedMonAdc2fc[j]+0.5);
        hfHists.shapeALL->Fill(i,tmp);
        hfHists.shapePED->Fill(i,tmp-calibs_.pedestal(digi.sample(i).capid()));
	vals[i] = tmp-calibs_.pedestal(digi.sample(i).capid());
      }
      if(doPerChannel_) perChanHists(3,digi.id(),vals,hfHists.shape, hfHists.time, hfHists.energy, baseFolder_);
    }
  return;

}

void HcalLEDMonitor::done(){
  return;
}

void HcalLEDMonitor::perChanHists(int id, const HcalDetId detid, float* vals, 
				  map<HcalDetId, MonitorElement*> &tShape, 
				  map<HcalDetId, MonitorElement*> &tTime,
				  map<HcalDetId, MonitorElement*> &tEnergy,
				  string baseFolder){
  
  MonitorElement* _me;
  if(m_dbe==NULL) return;
  _meIter=tShape.begin();

  string type = "HB";
  if(id==1) type = "HE"; 
  else if(id==2) type = "HO"; 
  else if(id==3) type = "HF"; 
  
  if(m_dbe) m_dbe->setCurrentFolder(baseFolder+"/"+type);

  
  _meIter = tShape.find(detid);
  if (_meIter!=tShape.end()){
    _me= _meIter->second;
    if(_me==NULL && fVerbosity>0) cout <<"HcalLEDAnalysis::perChanHists  This histo is NULL!!??"<<endl;
    else{
      float en=0;
      float ts =0; float bs=0;
      int maxi=0; float maxa=0;
      for(int i=sigS0_; i<=sigS1_; i++){
	if(vals[i]>maxa){maxa=vals[i]; maxi=i;}
      }
      for(int i=sigS0_; i<=sigS1_; i++){	  
	en += vals[i];
	if(i>=(maxi-1) && i<=maxi+1){
	  ts += i*vals[i];
	  bs += vals[i];
	}
	_me->Fill(i,vals[i]);
      }
      _me = tTime[detid];      
      if(bs!=0) _me->Fill(ts/bs);
      _me = tEnergy[detid];  
      _me->Fill(en); 
    }
  }
  else{
    char name[1024];
    sprintf(name,"%s LED Shape ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert1;
    insert1 =  m_dbe->book1D(name,name,10,-0.5,9.5);
    float en=0;
    float ts =0; float bs=0;
    int maxi=0; float maxa=0;
    for(int i=sigS0_; i<=sigS1_; i++){
      if(vals[i]>maxa){maxa=vals[i]; maxi=i;}
      insert1->Fill(i,vals[i]); 
    }
    for(int i=sigS0_; i<=sigS1_; i++){	  
      en += vals[i];
      if(i>=(maxi-1) && i<=maxi+1){
	ts += i*vals[i];
	bs += vals[i];
      }
    }
    tShape[detid] = insert1;
    
    sprintf(name,"%s LED Time ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert2 =  m_dbe->book1D(name,name,100,0,10);
    if(bs!=0) insert2->Fill(ts/bs); 
    tTime[detid] = insert2;	

    sprintf(name,"%s LED Energy ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert3 =  m_dbe->book1D(name,name,250,0,5000);
    insert3->Fill(en); 
    tEnergy[detid] = insert3;	
    
  } 

  return;

}  
