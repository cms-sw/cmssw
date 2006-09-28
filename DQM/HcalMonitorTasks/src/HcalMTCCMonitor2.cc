#include "DQM/HcalMonitorTasks/interface/HcalMTCCMonitor2.h"

HcalMTCCMonitor2::HcalMTCCMonitor2() {
  occThresh_ = 1.0;
  ievt_=0;
}

HcalMTCCMonitor2::~HcalMTCCMonitor2() {}

void HcalMTCCMonitor2::clearME(){
   if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2/HO");
    m_dbe->removeContents(); 
    meEVT_= 0;
  }
}
void HcalMTCCMonitor2::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "MTCC eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "MTCC phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;
  
  occThresh_ = ps.getUntrackedParameter<double>("MTCCOccThresh", 10.0);
  cout << "MTCC occupancy threshold set to " << occThresh_ << endl;
  
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2");
    meTrig_  = m_dbe->book1D("LTC Trigger","LTC Trigger",6,0,5);

    meEVT_ = m_dbe->bookInt("MTCC Event Number");    
    meEVT_->Fill(ievt_);
 
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2/HB");
    hbP.DT = m_dbe->book1D("HB Top DT Trigger Time","HB Top DT Trigger Time",100,0,9);
    hbP.CSC = m_dbe->book1D("HB Top CSC Trigger Time","HB Top CSC Trigger Time",100,0,9);
    hbP.RBC1 = m_dbe->book1D("HB Top RBC1 Trigger Time","HB Top RBC1 Trigger Time",100,0,9);
    hbP.RBC2 = m_dbe->book1D("HB Top RBC2 Trigger Time","HB Top RBC2 Trigger Time",100,0,9);
    hbP.RBCTB = m_dbe->book1D("HB Top RBCTB Trigger Time","HB Top RBCTB Trigger Time",100,0,9);
    hbP.NA = m_dbe->book1D("HB Top NA Trigger Time","HB Top NA Trigger Time",100,0,9);
    hbP.GLTRIG = m_dbe->book1D("HB Top All Triggers Time","HB Top All Triggers Time",150,-150,150);
    hbP.E = m_dbe->book1D("HB Top Hit Energy","HB Top Hit Energy",200,0,50);
    hbP.PEDS = m_dbe->book1D("HB Top Ped Vals","HB Top Ped Vals",100,0,20);
    hbP.OCC = m_dbe->book2D("HB Top Geo Occupancy Map","HB Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hbM.DT = m_dbe->book1D("HB Bottom DT Trigger Time","HB Bottom DT Trigger Time",100,0,9);
    hbM.CSC = m_dbe->book1D("HB Bottom CSC Trigger Time","HB Bottom CSC Trigger Time",100,0,9);
    hbM.RBC1 = m_dbe->book1D("HB Bottom RBC1 Trigger Time","HB Bottom RBC1 Trigger Time",100,0,9);
    hbM.RBC2 = m_dbe->book1D("HB Bottom RBC2 Trigger Time","HB Bottom RBC2 Trigger Time",100,0,9);
    hbM.RBCTB = m_dbe->book1D("HB Bottom RBCTB Trigger Time","HB Bottom RBCTB Trigger Time",100,0,9);
    hbM.NA = m_dbe->book1D("HB Bottom NA Trigger Time","HB Bottom NA Trigger Time",100,0,9);
    hbM.GLTRIG = m_dbe->book1D("HB Bottom All Triggers Time","HB Bottom All Triggers Time",150,-150,150);
    hbM.E = m_dbe->book1D("HB Bottom Hit Energy","HB Bottom Hit Energy",100,0,50);
    hbM.PEDS = m_dbe->book1D("HB Bottom Ped Vals","HB Bottom Ped Vals",100,0,20);
    hbM.OCC = m_dbe->book2D("HB Bottom Geo Occupancy Map","HB Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2/HE");
    heP.DT = m_dbe->book1D("HE Top DT Trigger Time","HE Top DT Trigger Time",100,0,9);
    heP.CSC = m_dbe->book1D("HE Top CSC Trigger Time","HE Top CSC Trigger Time",100,0,9);
    heP.RBC1 = m_dbe->book1D("HE Top RBC1 Trigger Time","HE Top RBC1 Trigger Time",100,0,9);
    heP.RBC2 = m_dbe->book1D("HE Top RBC2 Trigger Time","HE Top RBC2 Trigger Time",100,0,9);
    heP.RBCTB = m_dbe->book1D("HE Top RBCTB Trigger Time","HE Top RBCTB Trigger Time",100,0,9);
    heP.NA = m_dbe->book1D("HE Top NA Trigger Time","HE Top NA Trigger Time",100,0,9);
    heP.GLTRIG = m_dbe->book1D("HE Top All Triggers Time","HE Top All Triggers Time",150,-150,150);
    heP.E = m_dbe->book1D("HE Top Hit Energy","HE Top Hit Energy",100,0,50);
    heP.PEDS = m_dbe->book1D("HE Top Ped Vals","HE Top Ped Vals",100,0,20);
    heP.OCC = m_dbe->book2D("HE Top Geo Occupancy Map","HE Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    heM.DT = m_dbe->book1D("HE Bottom DT Trigger Time","HE Bottom DT Trigger Time",100,0,9);
    heM.CSC = m_dbe->book1D("HE Bottom CSC Trigger Time","HE Bottom CSC Trigger Time",100,0,9);
    heM.RBC1 = m_dbe->book1D("HE Bottom RBC1 Trigger Time","HE Bottom RBC1 Trigger Time",100,0,9);
    heM.RBC2 = m_dbe->book1D("HE Bottom RBC2 Trigger Time","HE Bottom RBC2 Trigger Time",100,0,9);
    heM.RBCTB = m_dbe->book1D("HE Bottom RBCTB Trigger Time","HE Bottom RBCTB Trigger Time",100,0,9);
    heM.NA = m_dbe->book1D("HE Bottom NA Trigger Time","HE Bottom NA Trigger Time",100,0,9);
    heM.GLTRIG = m_dbe->book1D("HE Bottom All Triggers Time","HE Bottom All Triggers Time",150,-150,150);
    heM.PEDS = m_dbe->book1D("HE Bottom Ped Vals","HE Bottom Ped Vals",100,0,20);
    heM.E = m_dbe->book1D("HE Bottom Hit Energy","HE Bottom Hit Energy",100,0,50);
    heM.OCC = m_dbe->book2D("HE Bottom Geo Occupancy Map","HE Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor2/HO");
    hoP1.DT = m_dbe->book1D("HO YB2 Top DT Trigger Time","HO YB2 Top DT Trigger Time",100,0,9);
    hoP1.CSC = m_dbe->book1D("HO YB2 Top CSC Trigger Time","HO YB2 Top CSC Trigger Time",100,0,9);
    hoP1.RBC1 = m_dbe->book1D("HO YB2 Top RBC1 Trigger Time","HO YB2 Top RBC1 Trigger Time",100,0,9);
    hoP1.RBC2 = m_dbe->book1D("HO YB2 Top RBC2 Trigger Time","HO YB2 Top RBC2 Trigger Time",100,0,9);
    hoP1.RBCTB = m_dbe->book1D("HO YB2 Top RBCTB Trigger Time","HO YB2 Top RBCTB Trigger Time",100,0,9);
    hoP1.NA = m_dbe->book1D("HO YB2 Top NA Trigger Time","HO YB2 Top NA Trigger Time",100,0,9);
    hoP1.GLTRIG = m_dbe->book1D("HO YB2 Top All Triggers Time","HO YB2 Top All Triggers Time",150,-150,150);
    hoP1.E = m_dbe->book1D("HO YB2 Top Hit Energy","HO YB2 Top Hit Energy",100,0,50);
    hoP1.PEDS = m_dbe->book1D("HO YB2 Top Ped Vals","HO YB2 Top Ped Vals",100,0,20);
    hoP1.OCC = m_dbe->book2D("HO YB2 Top Geo Occupancy Map","HO YB2 Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hoM1.DT = m_dbe->book1D("HO YB2 Bottom DT Trigger Time","HO YB2 Bottom DT Trigger Time",100,0,9);
    hoM1.CSC = m_dbe->book1D("HO YB2 Bottom CSC Trigger Time","HO YB2 Bottom CSC Trigger Time",100,0,9);
    hoM1.RBC1 = m_dbe->book1D("HO YB2 Bottom RBC1 Trigger Time","HO YB2 Bottom RBC1 Trigger Time",100,0,9);
    hoM1.RBC2 = m_dbe->book1D("HO YB2 Bottom RBC2 Trigger Time","HO YB2 Bottom RBC2 Trigger Time",100,0,9);
    hoM1.RBCTB = m_dbe->book1D("HO YB2 Bottom RBCTB Trigger Time","HO YB2 Bottom RBCTB Trigger Time",100,0,9);
    hoM1.NA = m_dbe->book1D("HO YB2 Bottom NA Trigger Time","HO YB2 Bottom NA Trigger Time",100,0,9);
    hoM1.GLTRIG = m_dbe->book1D("HO YB2 Bottom All Triggers Time","HO YB2 Bottom All Triggers Time",150,-150,150);
    hoM1.E = m_dbe->book1D("HO YB2 Bottom Hit Energy","HO YB2 Bottom Hit Energy",100,0,50);
    hoM1.PEDS = m_dbe->book1D("HO YB2 Bottom Ped Vals","HO YB2 Bottom Ped Vals",100,0,20);
    hoM1.OCC = m_dbe->book2D("HO YB2 Bottom Geo Occupancy Map","HO YB2 Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    hoP2.DT = m_dbe->book1D("HO YB1/0 Top DT Trigger Time","HO YB1/0 Top DT Trigger Time",100,0,9);
    hoP2.CSC = m_dbe->book1D("HO YB1/0 Top CSC Trigger Time","HO YB1/0 Top CSC Trigger Time",100,0,9);
    hoP2.RBC1 = m_dbe->book1D("HO YB1/0 Top RBC1 Trigger Time","HO YB1/0 Top RBC1 Trigger Time",100,0,9);
    hoP2.RBC2 = m_dbe->book1D("HO YB1/0 Top RBC2 Trigger Time","HO YB1/0 Top RBC2 Trigger Time",100,0,9);
    hoP2.RBCTB = m_dbe->book1D("HO YB1/0 Top RBCTB Trigger Time","HO YB1/0 Top RBCTB Trigger Time",100,0,9);
    hoP2.NA = m_dbe->book1D("HO YB1/0 Top NA Trigger Time","HO YB1/0 Top NA Trigger Time",100,0,9);
    hoP2.GLTRIG = m_dbe->book1D("HO YB1/0 Top All Triggers Time","HO YB1/0 Top All Triggers Time",150,-150,150);
    hoP2.E = m_dbe->book1D("HO YB1/0 Top Hit Energy","HO YB1/0 Top Hit Energy",100,0,50);
    hoP2.PEDS = m_dbe->book1D("HO YB1/0 Top Ped Vals","HO YB1/0 Top Ped Vals",100,0,20);
    hoP2.OCC = m_dbe->book2D("HO YB1/0 Top Geo Occupancy Map","HO YB1/0 Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hoM2.DT = m_dbe->book1D("HO YB1/0 Bottom DT Trigger Time","HO YB1/0 Bottom DT Trigger Time",100,0,9);
    hoM2.CSC = m_dbe->book1D("HO YB1/0 Bottom CSC Trigger Time","HO YB1/0 Bottom CSC Trigger Time",100,0,9);
    hoM2.RBC1 = m_dbe->book1D("HO YB1/0 Bottom RBC1 Trigger Time","HO YB1/0 Bottom RBC1 Trigger Time",100,0,9);
    hoM2.RBC2 = m_dbe->book1D("HO YB1/0 Bottom RBC2 Trigger Time","HO YB1/0 Bottom RBC2 Trigger Time",100,0,9);
    hoM2.RBCTB = m_dbe->book1D("HO YB1/0 Bottom RBCTB Trigger Time","HO YB1/0 Bottom RBCTB Trigger Time",100,0,9);
    hoM2.NA = m_dbe->book1D("HO YB1/0 Bottom NA Trigger Time","HO YB1/0 Bottom NA Trigger Time",100,0,9);
    hoM2.GLTRIG = m_dbe->book1D("HO YB1/0 Bottom All Triggers Time","HO YB1/0 Bottom All Triggers Time",150,-150,150);
    hoP2.PEDS = m_dbe->book1D("HO YB1/0 Bottom Ped Vals","HO YB1/0 Bottom Ped Vals",100,0,20);
    hoM2.E = m_dbe->book1D("HO YB1/0 Bottom Hit Energy","HO YB1/0 Bottom Hit Energy",100,0,50);
    hoM2.OCC = m_dbe->book2D("HO YB1/0 Bottom Geo Occupancy Map","HO YB1/0 Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


  }

  return;
}

void HcalMTCCMonitor2::processEvent(const HBHEDigiCollection& hbhe,
				    const HODigiCollection& ho,
				    const LTCDigiCollection& ltc,
				    const HcalDbService& cond){
  
  if(!m_dbe) { printf("HcalMTCCMonitor2::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }
  
  ievt_++;
  meEVT_->Fill(ievt_);

  if(ltc.size()<1) return;
  
  LTCDigi trig; 
  LTCDigiCollection::const_iterator digiItr = ltc.begin();
  trig = *digiItr;

  // get conditions
  const HcalQIEShape* shape = cond.getHcalShape(); // this one is generic  
  HcalCalibrations calibs;

  if ( m_dbe !=NULL ) {
    
    for(int t = 0; t<6; t++){
      if(trig.HasTriggered(t)) meTrig_->Fill(t);
    }
    
    try{      
      for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	

	cond.makeHcalCalibration (digi.id(), &calibs);
	const HcalQIECoder* channelCoder = cond.getHcalCoder(digi.id());
	HcalCoderDb coder(*channelCoder, *shape);
	CaloSamples tool;
	coder.adc2fC(digi,tool);
	
	double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
	double fc_ampl=0; double mtime = -1;
	for (int i=0; i<tool.size(); i++) {
	  int capid=digi[i].capid();
	  ta = (tool[i]-calibs.pedestal(capid)); // pedestal subtraction
	  fc_ampl+=ta; 
	  ta*=calibs.gain(capid); // fC --> GeV
	  if(ta>maxA){
	    maxA=ta;
	    maxI=i;
	  }
	}

	if(fc_ampl>(occThresh_*1.5)){
	  double m1 = 0, z=0, p1=0;
	  int capid=0;
	  if(maxI!=0){
	    capid=digi[maxI-1].capid();
	    m1 = (tool[maxI-1]-calibs.pedestal(capid))*calibs.gain(capid);
	  }
	  capid=digi[maxI-1].capid();
	  z = (tool[maxI]-calibs.pedestal(capid))*calibs.gain(capid);
	  
	  if(maxI!=(tool.size()-1)){
	    capid=digi[maxI+1].capid();
	    p1 = (tool[maxI+1]-calibs.pedestal(capid))*calibs.gain(capid);
	  }
	  
	  mtime = m1*(maxI-1) + z*maxI + p1*(maxI+1);
	  mtime /= (m1 + z + p1);
	  
	  if(digi.id().iphi()<31){	    
	    if(digi.id().subdet()==HcalBarrel){
	      hbP.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      hbP.GLTRIG->Fill(mtime);
	      hbP.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) hbP.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) hbP.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) hbP.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) hbP.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) hbP.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) hbP.NA->Fill(mtime);
	    }  //barrel
	    else if(digi.id().subdet()==HcalEndcap){	    
	      heP.OCC->Fill(digi.id().ieta(),digi.id().iphi());	
	      heP.GLTRIG->Fill(mtime);
	      heP.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) heP.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) heP.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) heP.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) heP.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) heP.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) heP.NA->Fill(mtime);
	    } //endcap
	  }//iphi
	  else{
	    if(digi.id().subdet()==HcalBarrel){
	      hbM.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      hbM.GLTRIG->Fill(mtime);
	      hbM.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) hbM.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) hbM.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) hbM.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) hbM.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) hbM.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) hbM.NA->Fill(mtime);
	    } //barrel
	    else if(digi.id().subdet()==HcalEndcap){	 
	      heM.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      heM.GLTRIG->Fill(mtime);
	      heM.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) heM.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) heM.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) heM.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) heM.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) heM.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) heM.NA->Fill(mtime);
	    }//endcap
	  } //iphi
	}//loop over digis
      }
    }catch (...) {
      
      printf("HcalMTCCMonitor2::processEvent  No HBHE Digis.\n");
    }
    
    
    try{      
      for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
	const HODataFrame digi = (const HODataFrame)(*j);	
	
	cond.makeHcalCalibration(digi.id(), &calibs);
	const HcalQIECoder* channelCoder = cond.getHcalCoder(digi.id());
	HcalCoderDb coder(*channelCoder, *shape);
	CaloSamples tool;
	coder.adc2fC(digi,tool);
	
	double ampl=0; int maxI = -1; double maxA = -1e10; float ta=0;
	double fc_ampl=0; double mtime = -1;
	for (int i=0; i<tool.size(); i++) {
	  int capid=digi[i].capid();
	  ta = (tool[i]-calibs.pedestal(capid)); // pedestal subtraction
	  fc_ampl+=ta; 
	  ta*=calibs.gain(capid); // fC --> GeV
	  if(ta>maxA){
	    maxA=ta;
	    maxI=i;
	  }
	}
	if(fc_ampl>(occThresh_*1.5)){
	  double m1 = 0, z=0, p1=0;
	  int capid=0;
	  if(maxI!=0){
	    capid=digi[maxI-1].capid();
	    m1 = (tool[maxI-1]-calibs.pedestal(capid))*calibs.gain(capid);
	  }
	  capid=digi[maxI-1].capid();
	  z = (tool[maxI]-calibs.pedestal(capid))*calibs.gain(capid);
	  
	  if(maxI!=(tool.size()-1)){
	    capid=digi[maxI+1].capid();
	    p1 = (tool[maxI+1]-calibs.pedestal(capid))*calibs.gain(capid);
	  }
	  
	  mtime = m1*(maxI-1) + z*maxI + p1*(maxI+1);
	  mtime /= (m1 + z + p1);
	  
	  if(digi.id().iphi()<30){
	    if(digi.id().ieta()>10){
	      hoP1.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      hoP1.GLTRIG->Fill(mtime);
	      hoP1.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) hoP1.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) hoP1.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) hoP1.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) hoP1.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) hoP1.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) hoP1.NA->Fill(mtime);
	    }//ieta
	    else{
	      hoP2.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      hoP2.GLTRIG->Fill(mtime);
	      hoP2.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) hoP2.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) hoP2.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) hoP2.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) hoP2.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) hoP2.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) hoP2.NA->Fill(mtime);
	    }//ieta
	  }//iphi
	  else{	    
	    if(digi.id().ieta()>10){
	      hoM1.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      hoM1.GLTRIG->Fill(mtime);
	      hoM1.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) hoM1.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) hoM1.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) hoM1.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) hoM1.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) hoM1.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) hoM1.NA->Fill(mtime);
	    }//ieta
	    else{
	      hoM2.OCC->Fill(digi.id().ieta(),digi.id().iphi());
	      hoM2.GLTRIG->Fill(mtime);
	      hoM2.E->Fill(fc_ampl);
	      if(trig.HasTriggered(0)) hoM2.DT->Fill(mtime);
	      if(trig.HasTriggered(1)) hoM2.CSC->Fill(mtime);
	      if(trig.HasTriggered(2)) hoM2.RBC1->Fill(mtime);
	      if(trig.HasTriggered(3)) hoM2.RBC2->Fill(mtime);
	      if(trig.HasTriggered(4)) hoM2.RBCTB->Fill(mtime);
	      if(trig.HasTriggered(5)) hoM2.NA->Fill(mtime);
	    }//ieta
	  }//iphi
	}//loop over digis
      }
    }catch (...) {
      printf("HcalMTCCMonitor2::processEvent  No HBHE Digis.\n");
    }
  }//if mdbe
      
  return;
}


//HB (here i define HB as depth=1 and eta <=15)
//HE (here i define HE as depth=2 and eta>16)
//HO(YB+2) (here i define HO/yb+2 as depth=4 and eta>10
//HO(yb+1,yb0) (depth=4 and eta<=10)
