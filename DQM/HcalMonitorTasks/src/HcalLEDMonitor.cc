#include "DQM/HcalMonitorTasks/interface/HcalLEDMonitor.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"

HcalLEDMonitor::HcalLEDMonitor() {m_doPerChannel = false;}

HcalLEDMonitor::~HcalLEDMonitor() {}

void HcalLEDMonitor::clearME(){
     if ( m_dbe ) {
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HBHE");
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
    m_doPerChannel = true;
  }
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);

  ievt_=0;

  if ( m_dbe ) {

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor");
    meEVT_ = m_dbe->bookInt("LED Task Event Number");    
    meEVT_->Fill(ievt_);

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HBHE");
    hbHists.shapePED =  m_dbe->book1D("HBHE Ped Subtracted Pulse Shape","HBHE Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hbHists.shapeALL =  m_dbe->book1D("HBHE Average Pulse Shape","HBHE Average Pulse Shape",10,-0.5,9.5);
    hbHists.timeALL =  m_dbe->book1D("HBHE Average Pulse Time","HBHE Average Pulse Time",200,0,4);
    hbHists.rms_ped =  m_dbe->book1D("HBHE LED Ped Region RMS Values","HBHE LED Ped Region RMS Values",100,0,2);
    hbHists.mean_ped =  m_dbe->book1D("HBHE LED Ped Region Mean Values","HBHE LED Ped Region Mean Values",100,0,4);
    hbHists.rms_sig =  m_dbe->book1D("HBHE LED Sig Region RMS Values","HBHE LED Sig Region RMS Values",100,0,2);
    hbHists.mean_sig =  m_dbe->book1D("HBHE LED Sig Region Mean Values","HBHE LED Sig Region Mean Values",100,4,8);
    hbHists.rms_tail =  m_dbe->book1D("HBHE LED Tail Region RMS Values","HBHE LED Tail Region RMS Values",100,0,2);
    hbHists.mean_tail =  m_dbe->book1D("HBHE LED Tail Region Mean Values","HBHE LED Tail Region Mean Values",100,8,10);
    hbHists.rms_time =  m_dbe->book1D("HBHE LED Time RMS Values","HBHE LED Time RMS Values",100,0,0.02);
    hbHists.mean_time =  m_dbe->book1D("HBHE LED Time Mean Values","HBHE LED Time Mean Values",100,4,8);
    hbHists.err_map_geo =  m_dbe->book2D("HBHE LED Geo Error Map","HBHE LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.err_map_elec =  m_dbe->book2D("HBHE LED Elec Error Map","HBHE LED Elec Error Map",20,0,20,20,0,20);
    
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HF");
    hfHists.shapePED =  m_dbe->book1D("HF Ped Subtracted Pulse Shape","HF Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hfHists.shapeALL =  m_dbe->book1D("HF Average Pulse Shape","HF Average Pulse Shape",10,-0.5,9.5);
    hfHists.timeALL =  m_dbe->book1D("HF Average Pulse Time","HF Average Pulse Time",200,0,4);
    hfHists.rms_ped =  m_dbe->book1D("HF LED Ped Region RMS Values","HF LED Ped Region RMS Values",100,0,2);
    hfHists.mean_ped =  m_dbe->book1D("HF LED Ped Region Mean Values","HF LED Ped Region Mean Values",100,0,4);
    hfHists.rms_sig =  m_dbe->book1D("HF LED Sig Region RMS Values","HF LED Sig Region RMS Values",100,0,2);
    hfHists.mean_sig =  m_dbe->book1D("HF LED Sig Region Mean Values","HF LED Sig Region Mean Values",100,4,8);
    hfHists.rms_tail =  m_dbe->book1D("HF LED Tail Region RMS Values","HF LED Tail Region RMS Values",100,0,2);
    hfHists.mean_tail =  m_dbe->book1D("HF LED Tail Region Mean Values","HF LED Tail Region Mean Values",100,8,10);
    hfHists.rms_time =  m_dbe->book1D("HF LED Time RMS Values","HF LED Time RMS Values",100,0,0.02);
    hfHists.mean_time =  m_dbe->book1D("HF LED Time Mean Values","HF LED Time Mean Values",100,4,8);
    hfHists.err_map_geo =  m_dbe->book2D("HF LED Geo Error Map","HF LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.err_map_elec =  m_dbe->book2D("HF LED Elec Error Map","HF LED Elec Error Map",20,0,20,20,0,20);

    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HO");
    hoHists.shapePED =  m_dbe->book1D("HO Ped Subtracted Pulse Shape","HO Ped Subtracted Pulse Shape",10,-0.5,9.5);
    hoHists.shapeALL =  m_dbe->book1D("HO Average Pulse Shape","HO Average Pulse Shape",10,-0.5,9.5);
    hoHists.timeALL =  m_dbe->book1D("HO Average Pulse Time","HO Average Pulse Time",200,0,4);
    hoHists.rms_ped =  m_dbe->book1D("HO LED Ped Region RMS Values","HO LED Ped Region RMS Values",100,0,2);
    hoHists.mean_ped =  m_dbe->book1D("HO LED Ped Region Mean Values","HO LED Ped Region Mean Values",100,0,4);
    hoHists.rms_sig =  m_dbe->book1D("HO LED Sig Region RMS Values","HO LED Sig Region RMS Values",100,0,2);
    hoHists.mean_sig =  m_dbe->book1D("HO LED Sig Region Mean Values","HO LED Sig Region Mean Values",100,4,8);
    hoHists.rms_tail =  m_dbe->book1D("HO LED Tail Region RMS Values","HO LED Tail Region RMS Values",100,0,2);
    hoHists.mean_tail =  m_dbe->book1D("HO LED Tail Region Mean Values","HO LED Tail Region Mean Values",100,8,10);
    hoHists.rms_time =  m_dbe->book1D("HO LED Time RMS Values","HO LED Time RMS Values",100,0,0.02);
    hoHists.mean_time =  m_dbe->book1D("HO LED Time Mean Values","HO LED Time Mean Values",100,4,8);
    hoHists.err_map_geo =  m_dbe->book2D("HO LED Geo Error Map","HO LED Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.err_map_elec =  m_dbe->book2D("HO LED Elec Error Map","HO LED Elec Error Map",20,0,20,20,0,20);
  }

  return;
}

void HcalLEDMonitor::processEvent(const HBHEDigiCollection& hbhe,
				  const HODigiCollection& ho,
				  const HFDigiCollection& hf){
  

  ievt_++;
  meEVT_->Fill(ievt_);

  if(!m_dbe) { printf("HcalLEDMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  try{
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      //      if(digi.size()<10) continue;
      float vals[10];
      float ped = (digi.sample(0).adc()+digi.sample(1).adc()+digi.sample(2).adc()+digi.sample(3).adc())/4.0;      
      float t = (digi.sample(4).adc()-ped
      		 +(digi.sample(5).adc()-ped)*2.0
      		 +(digi.sample(6).adc()-ped)*3.0
      		 +(digi.sample(7).adc()-ped)*4.0);
      float b = (digi.sample(4).adc()+digi.sample(5).adc()+digi.sample(6).adc()+digi.sample(7).adc()-4.0*ped);
      if(b!=0) hbHists.timeALL->Fill(t/b);
      for (int i=0; i<digi.size(); i++) {
	if(i<10){
	  hbHists.shapeALL->Fill(i,digi.sample(i).adc());
	  hbHists.shapePED->Fill(i,digi.sample(i).adc()-ped);
	  vals[i] = digi.sample(i).adc()-ped;
	}
      }
      if(m_doPerChannel) perChanHists(0,digi.id(),vals,hbHists.shape, hbHists.pedRange, 
				      hbHists.sigRange, hbHists.tailRange, hbHists.time);

    }
  } catch (...) {
    printf("HcalLEDMonitor::processEvent  No HBHE Digis.\n");
  }
  
  try{
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);	
      if(digi.size()!=10) continue;
      float vals[10];
      float ped = (digi.sample(0).adc()+digi.sample(1).adc()+digi.sample(2).adc()+digi.sample(3).adc())/4.0;      
      float t = (digi.sample(4).adc()-ped
      		 +(digi.sample(5).adc()-ped)*2.0
      		 +(digi.sample(6).adc()-ped)*3.0
      		 +(digi.sample(7).adc()-ped)*4.0);
      float b = (digi.sample(4).adc()+digi.sample(5).adc()+digi.sample(6).adc()+digi.sample(7).adc()-4.0*ped);
      if(b!=0) hoHists.timeALL->Fill(t/b);
      for (int i=0; i<digi.size(); i++) {
	hoHists.shapeALL->Fill(i,digi.sample(i).adc());
	hoHists.shapePED->Fill(i,digi.sample(i).adc()-ped);
	vals[i] = digi.sample(i).adc()-ped;
      }
      if(m_doPerChannel) perChanHists(1,digi.id(),vals,hoHists.shape, hoHists.pedRange, 
				      hoHists.sigRange, hoHists.tailRange, hoHists.time);
    }        
  } catch (...) {
    cout << "HcalLEDMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);	
      if(digi.size()<10) continue;
      float vals[10];
      float ped = (digi.sample(0).adc()+digi.sample(1).adc()+digi.sample(2).adc()+digi.sample(3).adc())/4.0;      
      float t = (digi.sample(4).adc()-ped
      		 +(digi.sample(5).adc()-ped)*2.0
      		 +(digi.sample(6).adc()-ped)*3.0
      		 +(digi.sample(7).adc()-ped)*4.0);
      float b = (digi.sample(4).adc()+digi.sample(5).adc()+digi.sample(6).adc()+digi.sample(7).adc()-4.0*ped);
      if(b!=0) hfHists.timeALL->Fill(t/b);
      for (int i=0; i<digi.size(); i++) {
	if(i<10){
	  hfHists.shapePED->Fill(i,digi.sample(i).adc()-ped);
	  hfHists.shapeALL->Fill(i,digi.sample(i).adc());
	  vals[i] = digi.sample(i).adc()-ped;
	}
      }
      if(m_doPerChannel) perChanHists(2,digi.id(),vals,hfHists.shape, hfHists.pedRange, 
				      hfHists.sigRange, hfHists.tailRange, hfHists.time);
    }
  } catch (...) {
    cout << "HcalLEDMonitor::processEvent  No HF Digis." << endl;
  }

  return;

}

void HcalLEDMonitor::done(){
  return;
}

void HcalLEDMonitor::perChanHists(int id, const HcalDetId detid, float* vals, map<HcalDetId, MonitorElement*> &tShape, 
				  map<HcalDetId, MonitorElement*> &tPed, map<HcalDetId, MonitorElement*> &tSig,
				  map<HcalDetId, MonitorElement*> &tTail,  map<HcalDetId, MonitorElement*> &tTime){
  
  MonitorElement* _me;
  if(m_dbe==NULL) return;

  _meo=tShape.begin();
  string type = "HBHE";
  m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HBHE");
  if(id==1){
    type = "HO";
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HO");
  }
  else if(id==2){
    type = "HF";
    m_dbe->setCurrentFolder("HcalMonitor/LEDMonitor/HF");
  }
  
  _meo = tShape.find(detid);
  if (_meo!=tShape.end()){
    _me= _meo->second;
    if(_me==NULL) printf("HcalLEDAnalysis::perChanHists  This histo is NULL!!??\n");
    else{
      for(int i=0; i<10; i++)
	_me->Fill(i,vals[i]);            
      _me = tPed[detid];
      for(int i=0; i<4; i++)
	_me->Fill(i,vals[i]); 
      _me = tSig[detid];
      for(int i=4; i<8; i++)
	_me->Fill(i,vals[i]);
      _me = tTail[detid];
      for(int i=8; i<10; i++)
	_me->Fill(i,vals[i]);	
      _me = tTime[detid];
      float t = (vals[4]+(vals[5])*2.0+(vals[6])*3.0+(vals[7])*4.0);
      float b = (vals[4]+vals[5]+vals[6]+vals[7]);
      if(b!=0) _me->Fill(t/b); 
    }
  }
  else{
    char name[1024];
    sprintf(name,"%s LED Shape ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert1;
    insert1 =  m_dbe->book1D(name,name,10,-0.5,9.5);
    for(int i=0; i<10; i++)
      insert1->Fill(i,vals[i]); 
    tShape[detid] = insert1;
    
    sprintf(name,"%s LED Pedestal Region ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());
    MonitorElement* insert2 =  m_dbe->book1D(name,name,5,0,4);
    for(int i=0; i<4; i++)
      insert2->Fill(i,vals[i]); 
    tPed[detid] = insert2;
    
    sprintf(name,"%s LED Signal Region ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());
    MonitorElement* insert3 =  m_dbe->book1D(name,name,5,4,8);
    for(int i=4; i<8; i++)
      insert3->Fill(i,vals[i]); 
    tSig[detid] = insert3;
    
    sprintf(name,"%s LED Tail Region ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());
    MonitorElement* insert4 =  m_dbe->book1D(name,name,3,8,10);
    for(int i=8; i<10; i++)
      insert4->Fill(i,vals[i]); 
    tTail[detid] = insert4;
    
    sprintf(name,"%s LED Time ieta=%d iphi=%d depth=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth());      
    MonitorElement* insert5 =  m_dbe->book1D(name,name,200,4,8);
    float t = (vals[4]*4.0+(vals[5])*5.0+(vals[6])*6.0+(vals[7])*7.0);
    float b = (vals[4]+vals[5]+vals[6]+vals[7]);
    if(b!=0) insert5->Fill(t/b); 
    tTime[detid] = insert5;	
    
  } 

  return;

}  
