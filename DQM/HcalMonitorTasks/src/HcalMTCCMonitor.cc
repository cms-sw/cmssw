#include "DQM/HcalMonitorTasks/interface/HcalMTCCMonitor.h"

HcalMTCCMonitor::HcalMTCCMonitor() {
  occThresh_ = 1.0;
  ievt_=0;
  shape_=NULL;
}

HcalMTCCMonitor::~HcalMTCCMonitor() {}

void HcalMTCCMonitor::clearME(){
   if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HO");
    m_dbe->removeContents(); 
    meEVT_= 0;
  }
}
void HcalMTCCMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "MTCC eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "MTCC phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;
  
  occThresh_ = ps.getUntrackedParameter<double>("MTCCOccThresh", 10);
  cout << "MTCC occupancy threshold set to " << occThresh_ << endl;
  
  dumpThresh_ = ps.getUntrackedParameter<double>("DumpThreshold", -1);
  dumpEtaLo_ = ps.getUntrackedParameter<double>("DumpEtaLow", -1);
  dumpEtaHi_ = ps.getUntrackedParameter<double>("DumpEtaHigh", -1);
  dumpPhiLo_ = ps.getUntrackedParameter<double>("DumpPhiLow", -1);
  dumpPhiHi_ = ps.getUntrackedParameter<double>("DumpPhiHigh", -1);

  ievt_=0;
  
  if ( m_dbe !=NULL ) {    
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor");
    meTrig_  = m_dbe->book1D("LTC Trigger","LTC Trigger",6,0,5);

    meEVT_ = m_dbe->bookInt("MTCC Event Number");    
    meEVT_->Fill(ievt_);
 
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HB");
    hbC.OCC = m_dbe->book2D("HB Geo Occupancy Map","HB Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbC.E = m_dbe->book1D("HB Hit Energy","HB Hit Energy",100,0,50);
    hbC.DT = m_dbe->book1D("HB DT Trigger Time","HB DT Trigger Time",100,0,9);

    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HE");
    heC.CSC = m_dbe->book1D("HE Bottom CSC Trigger Time","HE Bottom CSC Trigger Time",100,0,9);
    heC.E = m_dbe->book1D("HE Bottom Hit Energy","HE Bottom Hit Energy",100,0,50);
    heC.OCC = m_dbe->book2D("HE Bottom Geo Occupancy Map","HE Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    

    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HO");
    hoP1.DT = m_dbe->book1D("HO YB2 Top DT Trigger Time","HO YB2 Top DT Trigger Time",100,0,9);
    hoM1.DT = m_dbe->book1D("HO YB2 Bottom DT Trigger Time","HO YB2 Bottom DT Trigger Time",100,0,9);
    hoP2.DT = m_dbe->book1D("HO YB1/0 Top DT Trigger Time","HO YB1/0 Top DT Trigger Time",100,0,9);
    hoM2.DT = m_dbe->book1D("HO YB1/0 Bottom DT Trigger Time","HO YB1/0 Bottom DT Trigger Time",100,0,9);
    hoC.OCC = m_dbe->book2D("HO YB Geo Occupancy Map","HO YB Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoC.E = m_dbe->book1D("HO YB Hit Energy","HO YB Hit Energy",100,0,50);

  }

  return;
}

void HcalMTCCMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const LTCDigiCollection& ltc,
				   const HcalDbService& cond){
  
  if(!m_dbe) { printf("HcalMTCCMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  // get conditions
  if(!shape_) shape_ = cond.getHcalShape(); // this one is generic  
  
  dumpDigi(hbhe, ho, cond);

  LTCDigi trig; 
  try{      
    if(ltc.size()<1) return;        
  }catch (...) {      
    printf("HcalMTCCMonitor::processEvent  No LTC Digi.\n"); return;
  }
  LTCDigiCollection::const_iterator digiItr = ltc.begin();    
  trig = *digiItr;

  if ( m_dbe !=NULL ) {
    
    for(int t = 0; t<6; t++){
      if(trig.HasTriggered(t)) meTrig_->Fill(t);
    }
    try{      
      for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	

	cond.makeHcalCalibration(digi.id(), &calibs_);
	const HcalQIECoder* channelCoder = cond.getHcalCoder(digi.id());
	HcalCoderDb coder(*channelCoder, *shape_);
	CaloSamples tool;
	coder.adc2fC(digi,tool);
	
	int maxI = -1; double maxA = -1e10; float ta=0;
	double fc_ampl=0; double mtime = -1;
	for (int i=0; i<tool.size(); i++) {
	  int capid=digi[i].capid();
	  ta = (tool[i]-calibs_.pedestal(capid)); // pedestal subtraction
	  fc_ampl+=ta; 
	  ta*=calibs_.gain(capid); // fC --> GeV
	  if(ta>maxA){
	    maxA=ta;
	    maxI=i;
	  }
	}
	if(fc_ampl<occThresh_) continue;

	double m1 = 0, z=0, p1=0;
	int capid=0;
	if(maxI!=0){
	  capid=digi[maxI-1].capid();
	  m1 = (tool[maxI-1]-calibs_.pedestal(capid))*calibs_.gain(capid);
	}
	capid=digi[maxI].capid();
	z = (tool[maxI]-calibs_.pedestal(capid))*calibs_.gain(capid);
	
	if(maxI!=(tool.size()-1)){
	  capid=digi[maxI+1].capid();
	  p1 = (tool[maxI+1]-calibs_.pedestal(capid))*calibs_.gain(capid);
	}
	
	mtime = m1*(maxI-1) + z*maxI + p1*(maxI+1);
	mtime /= (m1 + z + p1);

	if(digi.id().iphi()<73){
	  if(digi.id().subdet()==HcalBarrel){
	    hbC.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	    hbC.E->Fill(fc_ampl);
            if(trig.HasTriggered(0)) hbC.DT->Fill(mtime);	    
	  }
	  else if(digi.id().subdet()==HcalEndcap){
	    heC.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	    heC.E->Fill(fc_ampl);	    
	    if(trig.HasTriggered(1)) heC.CSC->Fill(mtime);
	  }
	}
      }//loop over digis
    }catch (...) {      
      printf("HcalMTCCMonitor::processEvent  No HBHE Digis.\n");
    }
    
    try{      
      for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
	const HODataFrame digi = (const HODataFrame)(*j);	
	
	cond.makeHcalCalibration(digi.id(), &calibs_);
	const HcalQIECoder* channelCoder = cond.getHcalCoder(digi.id());
	HcalCoderDb coder(*channelCoder, *shape_);
	CaloSamples tool;
	coder.adc2fC(digi,tool);
	
	int maxI = -1; double maxA = -1e10; float ta=0;
	double fc_ampl=0; double mtime = -1;
	for (int i=0; i<tool.size(); i++) {
	  int capid=digi[i].capid();
	  ta = (tool[i]-calibs_.pedestal(capid)); // pedestal subtraction
	  fc_ampl+=ta; 
	  ta*=calibs_.gain(capid); // fC --> GeV
	  if(ta>maxA){
	    maxA=ta;
	    maxI=i;
	  }
	}
	if(fc_ampl<occThresh_) continue;

	double m1 = 0, z=0, p1=0;
	int capid=0;
	if(maxI!=0){
	  capid=digi[maxI-1].capid();
	  m1 = (tool[maxI-1]-calibs_.pedestal(capid))*calibs_.gain(capid);
	}
	capid=digi[maxI].capid();
	z = (tool[maxI]-calibs_.pedestal(capid))*calibs_.gain(capid);
	
	if(maxI!=(tool.size()-1)){
	  capid=digi[maxI+1].capid();
	  p1 = (tool[maxI+1]-calibs_.pedestal(capid))*calibs_.gain(capid);
	}
	
	mtime = m1*(maxI-1) + z*maxI + p1*(maxI+1);
	mtime /= (m1 + z + p1);

	if(digi.id().iphi()<73){
	  if(digi.id().ieta()>-5){
	    hoC.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	    hoC.E->Fill(fc_ampl);
	  }
	}

	if(digi.id().iphi()<30){
	  if(digi.id().ieta()>10){
	    if(trig.HasTriggered(0)) hoP1.DT->Fill(mtime);
	  }//ieta
	  else{
	    if(trig.HasTriggered(0)) hoP2.DT->Fill(mtime);
	  }//ieta
	}//iphi
	else{	    
	  if(digi.id().ieta()>10){
	    if(trig.HasTriggered(0)) hoM1.DT->Fill(mtime);
	  }//ieta
	  else{
	    if(trig.HasTriggered(0)) hoM2.DT->Fill(mtime);
	  }//ieta
	}//iphi
      }//loop over digis   
    }catch (...) {
      printf("HcalMTCCMonitor::processEvent  No HBHE Digis.\n");
    }
  }//if mdbe
  return;
}


void HcalMTCCMonitor::dumpDigi(const HBHEDigiCollection& hbhe, const HODigiCollection& ho, const HcalDbService& cond){
  if(dumpThresh_<0) return;

  float fc_ampl = 0;
  float ta = 0;  
  int myPhi = -1;
  try{      
    bool done = false;
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end() && !done; j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	
      if(digi.id().ieta()>dumpEtaHi_) continue;
      if(digi.id().iphi()>dumpPhiHi_) continue;
      if(digi.id().ieta()<dumpEtaLo_) continue;
      if(digi.id().iphi()<dumpPhiLo_) continue;

      cond.makeHcalCalibration(digi.id(), &calibs_);
      const HcalQIECoder* channelCoder = cond.getHcalCoder(digi.id());
      HcalCoderDb coder(*channelCoder, *shape_);
      CaloSamples tool;
      coder.adc2fC(digi,tool);

      fc_ampl=0;
      for (int i=0; i<tool.size(); i++) {
	int capid=digi[i].capid();
	ta = (tool[i]-calibs_.pedestal(capid)); // pedestal subtraction
	fc_ampl+=ta; 
      }
      if(fc_ampl>dumpThresh_){done = true; myPhi = digi.id().iphi(); }
    }
  }catch (...) {
    printf("HcalMTCCMonitor::processEvent  No HBHE Digis.\n");
  }

  if(fc_ampl>dumpThresh_){
    try{      
      for(int ieta = dumpEtaLo_; ieta<=dumpEtaHi_; ieta++){
	for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
	  const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	
	  if(digi.id().ieta()!=ieta) continue;
	  if(digi.id().iphi()!=myPhi) continue;
	  
	  cond.makeHcalCalibration(digi.id(), &calibs_);
	  const HcalQIECoder* channelCoder = cond.getHcalCoder(digi.id());
	  HcalCoderDb coder(*channelCoder, *shape_);
	  CaloSamples tool;
	  coder.adc2fC(digi,tool);
	  
	  printf("iPhi: %d, iEta: %d, BX ampl:",myPhi,ieta);
	  for (int i=0; i<tool.size(); i++) {
	    int capid=digi[i].capid();
	    ta = (tool[i]-calibs_.pedestal(capid)); // pedestal subtraction
	    printf(" %.3f,",ta);
	  }
	  printf("\n");
	}
      }      
    }catch (...) {
      printf("HcalMTCCMonitor::processEvent  No HBHE Digis.\n");
    }
  }

}


//HB (here i define HB as depth=1 and eta <=15)
//HE (here i define HE as depth=2 and eta>16)
//HO(YB+2) (here i define HO/yb+2 as depth=4 and eta>10
//HO(yb+1,yb0) (depth=4 and eta<=10)
