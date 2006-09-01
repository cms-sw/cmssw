#include "DQM/HcalMonitorTasks/interface/HcalMTCCMonitor.h"

HcalMTCCMonitor::HcalMTCCMonitor() {
  occThresh_ = 1.0;
  ievt_=0;
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
  
  occThresh_ = ps.getUntrackedParameter<double>("MTCCOccThresh", 1.0);
  cout << "MTCC occupancy threshold set to " << occThresh_ << endl;
  
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor");
    meTrig_  = m_dbe->book1D("LTC Trigger","LTC Trigger",6,0,5);

    meEVT_ = m_dbe->bookInt("MTCC Event Number");    
    meEVT_->Fill(ievt_);
 
    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HB");
    hbP.DT = m_dbe->book1D("HB Top DT Trigger Time","HB Top DT Trigger Time",150,-150,150);
    hbP.CSC = m_dbe->book1D("HB Top CSC Trigger Time","HB Top CSC Trigger Time",150,-150,150);
    hbP.RBC1 = m_dbe->book1D("HB Top RBC1 Trigger Time","HB Top RBC1 Trigger Time",150,-150,150);
    hbP.RBC2 = m_dbe->book1D("HB Top RBC2 Trigger Time","HB Top RBC2 Trigger Time",150,-150,150);
    hbP.RBCTB = m_dbe->book1D("HB Top RBCTB Trigger Time","HB Top RBCTB Trigger Time",150,-150,150);
    hbP.NA = m_dbe->book1D("HB Top NA Trigger Time","HB Top NA Trigger Time",150,-150,150);
    hbP.E = m_dbe->book1D("HB Top Hit Energy","HB Top Hit Energy",100,0,50);
    hbP.Etot = m_dbe->book1D("HB Top Total Energy","HB Top Total Energy",100,0,100);
    hbP.OCC = m_dbe->book2D("HB Top Geo Occupancy Map","HB Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hbM.DT = m_dbe->book1D("HB Bottom DT Trigger Time","HB Bottom DT Trigger Time",150,-150,150);
    hbM.CSC = m_dbe->book1D("HB Bottom CSC Trigger Time","HB Bottom CSC Trigger Time",150,-150,150);
    hbM.RBC1 = m_dbe->book1D("HB Bottom RBC1 Trigger Time","HB Bottom RBC1 Trigger Time",150,-150,150);
    hbM.RBC2 = m_dbe->book1D("HB Bottom RBC2 Trigger Time","HB Bottom RBC2 Trigger Time",150,-150,150);
    hbM.RBCTB = m_dbe->book1D("HB Bottom RBCTB Trigger Time","HB Bottom RBCTB Trigger Time",150,-150,150);
    hbM.NA = m_dbe->book1D("HB Bottom NA Trigger Time","HB Bottom NA Trigger Time",150,-150,150);
    hbM.E = m_dbe->book1D("HB Bottom Hit Energy","HB Bottom Hit Energy",100,0,50);
    hbM.Etot = m_dbe->book1D("HB Bottom Total Energy","HB Bottom Total Energy",100,0,100);
    hbM.OCC = m_dbe->book2D("HB Bottom Geo Occupancy Map","HB Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HE");
    heP.DT = m_dbe->book1D("HE Top DT Trigger Time","HE Top DT Trigger Time",150,-150,150);
    heP.CSC = m_dbe->book1D("HE Top CSC Trigger Time","HE Top CSC Trigger Time",150,-150,150);
    heP.RBC1 = m_dbe->book1D("HE Top RBC1 Trigger Time","HE Top RBC1 Trigger Time",150,-150,150);
    heP.RBC2 = m_dbe->book1D("HE Top RBC2 Trigger Time","HE Top RBC2 Trigger Time",150,-150,150);
    heP.RBCTB = m_dbe->book1D("HE Top RBCTB Trigger Time","HE Top RBCTB Trigger Time",150,-150,150);
    heP.NA = m_dbe->book1D("HE Top NA Trigger Time","HE Top NA Trigger Time",150,-150,150);
    heP.E = m_dbe->book1D("HE Top Hit Energy","HE Top Hit Energy",100,0,50);
    heP.Etot = m_dbe->book1D("HE Top Total Energy","HE Top Total Energy",100,0,100);
    heP.OCC = m_dbe->book2D("HE Top Geo Occupancy Map","HE Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    heM.DT = m_dbe->book1D("HE Bottom DT Trigger Time","HE Bottom DT Trigger Time",150,-150,150);
    heM.CSC = m_dbe->book1D("HE Bottom CSC Trigger Time","HE Bottom CSC Trigger Time",150,-150,150);
    heM.RBC1 = m_dbe->book1D("HE Bottom RBC1 Trigger Time","HE Bottom RBC1 Trigger Time",150,-150,150);
    heM.RBC2 = m_dbe->book1D("HE Bottom RBC2 Trigger Time","HE Bottom RBC2 Trigger Time",150,-150,150);
    heM.RBCTB = m_dbe->book1D("HE Bottom RBCTB Trigger Time","HE Bottom RBCTB Trigger Time",150,-150,150);
    heM.NA = m_dbe->book1D("HE Bottom NA Trigger Time","HE Bottom NA Trigger Time",150,-150,150);
    heM.E = m_dbe->book1D("HE Bottom Hit Energy","HE Bottom Hit Energy",100,0,50);
    heM.Etot = m_dbe->book1D("HE Bottom Total Energy","HE Bottom Total Energy",100,0,100);
    heM.OCC = m_dbe->book2D("HE Bottom Geo Occupancy Map","HE Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/MTCCMonitor/HO");
    hoP1.DT = m_dbe->book1D("HO YB2 Top DT Trigger Time","HO YB2 Top DT Trigger Time",150,-150,150);
    hoP1.CSC = m_dbe->book1D("HO YB2 Top CSC Trigger Time","HO YB2 Top CSC Trigger Time",150,-150,150);
    hoP1.RBC1 = m_dbe->book1D("HO YB2 Top RBC1 Trigger Time","HO YB2 Top RBC1 Trigger Time",150,-150,150);
    hoP1.RBC2 = m_dbe->book1D("HO YB2 Top RBC2 Trigger Time","HO YB2 Top RBC2 Trigger Time",150,-150,150);
    hoP1.RBCTB = m_dbe->book1D("HO YB2 Top RBCTB Trigger Time","HO YB2 Top RBCTB Trigger Time",150,-150,150);
    hoP1.NA = m_dbe->book1D("HO YB2 Top NA Trigger Time","HO YB2 Top NA Trigger Time",150,-150,150);
    hoP1.E = m_dbe->book1D("HO YB2 Top Hit Energy","HO YB2 Top Hit Energy",100,0,50);
    hoP1.Etot = m_dbe->book1D("HO YB2 Top Total Energy","HO YB2 Top Total Energy",100,0,100);
    hoP1.OCC = m_dbe->book2D("HO YB2 Top Geo Occupancy Map","HO YB2 Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hoM1.DT = m_dbe->book1D("HO YB2 Bottom DT Trigger Time","HO YB2 Bottom DT Trigger Time",150,-150,150);
    hoM1.CSC = m_dbe->book1D("HO YB2 Bottom CSC Trigger Time","HO YB2 Bottom CSC Trigger Time",150,-150,150);
    hoM1.RBC1 = m_dbe->book1D("HO YB2 Bottom RBC1 Trigger Time","HO YB2 Bottom RBC1 Trigger Time",150,-150,150);
    hoM1.RBC2 = m_dbe->book1D("HO YB2 Bottom RBC2 Trigger Time","HO YB2 Bottom RBC2 Trigger Time",150,-150,150);
    hoM1.RBCTB = m_dbe->book1D("HO YB2 Bottom RBCTB Trigger Time","HO YB2 Bottom RBCTB Trigger Time",150,-150,150);
    hoM1.NA = m_dbe->book1D("HO YB2 Bottom NA Trigger Time","HO YB2 Bottom NA Trigger Time",150,-150,150);
    hoM1.E = m_dbe->book1D("HO YB2 Bottom Hit Energy","HO YB2 Bottom Hit Energy",100,0,50);
    hoM1.Etot = m_dbe->book1D("HO YB2 Bottom Total Energy","HO YB2 Bottom Total Energy",100,0,100);
    hoM1.OCC = m_dbe->book2D("HO YB2 Bottom Geo Occupancy Map","HO YB2 Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    hoP2.DT = m_dbe->book1D("HO YB1/0 Top DT Trigger Time","HO YB1/0 Top DT Trigger Time",150,-150,150);
    hoP2.CSC = m_dbe->book1D("HO YB1/0 Top CSC Trigger Time","HO YB1/0 Top CSC Trigger Time",150,-150,150);
    hoP2.RBC1 = m_dbe->book1D("HO YB1/0 Top RBC1 Trigger Time","HO YB1/0 Top RBC1 Trigger Time",150,-150,150);
    hoP2.RBC2 = m_dbe->book1D("HO YB1/0 Top RBC2 Trigger Time","HO YB1/0 Top RBC2 Trigger Time",150,-150,150);
    hoP2.RBCTB = m_dbe->book1D("HO YB1/0 Top RBCTB Trigger Time","HO YB1/0 Top RBCTB Trigger Time",150,-150,150);
    hoP2.NA = m_dbe->book1D("HO YB1/0 Top NA Trigger Time","HO YB1/0 Top NA Trigger Time",150,-150,150);
    hoP2.E = m_dbe->book1D("HO YB1/0 Top Hit Energy","HO YB1/0 Top Hit Energy",100,0,50);
    hoP2.Etot = m_dbe->book1D("HO YB1/0 Top Total Energy","HO YB1/0 Top Total Energy",100,0,100);
    hoP2.OCC = m_dbe->book2D("HO YB1/0 Top Geo Occupancy Map","HO YB1/0 Top Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    hoM2.DT = m_dbe->book1D("HO YB1/0 Bottom DT Trigger Time","HO YB1/0 Bottom DT Trigger Time",150,-150,150);
    hoM2.CSC = m_dbe->book1D("HO YB1/0 Bottom CSC Trigger Time","HO YB1/0 Bottom CSC Trigger Time",150,-150,150);
    hoM2.RBC1 = m_dbe->book1D("HO YB1/0 Bottom RBC1 Trigger Time","HO YB1/0 Bottom RBC1 Trigger Time",150,-150,150);
    hoM2.RBC2 = m_dbe->book1D("HO YB1/0 Bottom RBC2 Trigger Time","HO YB1/0 Bottom RBC2 Trigger Time",150,-150,150);
    hoM2.RBCTB = m_dbe->book1D("HO YB1/0 Bottom RBCTB Trigger Time","HO YB1/0 Bottom RBCTB Trigger Time",150,-150,150);
    hoM2.NA = m_dbe->book1D("HO YB1/0 Bottom NA Trigger Time","HO YB1/0 Bottom NA Trigger Time",150,-150,150);
    hoM2.E = m_dbe->book1D("HO YB1/0 Bottom Hit Energy","HO YB1/0 Bottom Hit Energy",100,0,50);
    hoM2.Etot = m_dbe->book1D("HO YB1/0 Bottom Total Energy","HO YB1/0 Bottom Total Energy",100,0,100);
    hoM2.OCC = m_dbe->book2D("HO YB1/0 Bottom Geo Occupancy Map","HO YB1/0 Bottom Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


  }

  return;
}

void HcalMTCCMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const LTCDigiCollection& ltc){
  
  if(!m_dbe) { printf("HcalMTCCMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }
  
  ievt_++;
  meEVT_->Fill(ievt_);

  if(ltc.size()<1) return;
  
  if ( m_dbe !=NULL ) {
    LTCDigi trig; 
    LTCDigiCollection::const_iterator digiItr = ltc.begin();
    trig = *digiItr;
    
    for(int t = 0; t<6; t++){
      if(trig.HasTriggered(t)) meTrig_->Fill(t);
    }
    
    HBHERecHitCollection::const_iterator _ib;
    HORecHitCollection::const_iterator _io;
    
    float hoP1tot=0, hoM1tot=0, hoP2tot=0,hoM2tot=0;
    if(hbHits.size()>0){      
      float hbPtot=0, hbMtot=0, hePtot=0,heMtot=0;
      for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits   	
	if(_ib->energy()>occThresh_){	  
	  if(_ib->id().iphi()<31){	    
	    if(_ib->id().subdet()==HcalBarrel){
	      hbPtot += _ib->energy();
	      hbP.OCC->Fill(_ib->id().ieta(),_ib->id().iphi());
	      hbP.E->Fill(_ib->energy());
	      if(trig.HasTriggered(0)) hbP.DT->Fill(_ib->time());
	      if(trig.HasTriggered(1)) hbP.CSC->Fill(_ib->time());
	      if(trig.HasTriggered(2)) hbP.RBC1->Fill(_ib->time());
	      if(trig.HasTriggered(3)) hbP.RBC2->Fill(_ib->time());
	      if(trig.HasTriggered(4)) hbP.RBCTB->Fill(_ib->time());
	      if(trig.HasTriggered(5)) hbP.NA->Fill(_ib->time());
	    }  //barrel
	    else if(_ib->id().subdet()==HcalEndcap){	    
	      hePtot += _ib->energy();
	      heP.OCC->Fill(_ib->id().ieta(),_ib->id().iphi());	
	      heP.E->Fill(_ib->energy());
	      if(trig.HasTriggered(0)) heP.DT->Fill(_ib->time());
	      if(trig.HasTriggered(1)) heP.CSC->Fill(_ib->time());
	      if(trig.HasTriggered(2)) heP.RBC1->Fill(_ib->time());
	      if(trig.HasTriggered(3)) heP.RBC2->Fill(_ib->time());
	      if(trig.HasTriggered(4)) heP.RBCTB->Fill(_ib->time());
	      if(trig.HasTriggered(5)) heP.NA->Fill(_ib->time());
	    } //endcap
	  }//iphi
	  else{
	    if(_ib->id().subdet()==HcalBarrel){
	      hbMtot += _ib->energy();
	      hbM.OCC->Fill(_ib->id().ieta(),_ib->id().iphi());
	      hbM.E->Fill(_ib->energy());
	      if(trig.HasTriggered(0)) hbM.DT->Fill(_ib->time());
	      if(trig.HasTriggered(1)) hbM.CSC->Fill(_ib->time());
	      if(trig.HasTriggered(2)) hbM.RBC1->Fill(_ib->time());
	      if(trig.HasTriggered(3)) hbM.RBC2->Fill(_ib->time());
	      if(trig.HasTriggered(4)) hbM.RBCTB->Fill(_ib->time());
	      if(trig.HasTriggered(5)) hbM.NA->Fill(_ib->time());
	    } //barrel
	    else if(_ib->id().subdet()==HcalEndcap){	 
	      heMtot += _ib->energy();
	      heM.OCC->Fill(_ib->id().ieta(),_ib->id().iphi());
	      heM.E->Fill(_ib->energy());
	      if(trig.HasTriggered(0)) heM.DT->Fill(_ib->time());
	      if(trig.HasTriggered(1)) heM.CSC->Fill(_ib->time());
	      if(trig.HasTriggered(2)) heM.RBC1->Fill(_ib->time());
	      if(trig.HasTriggered(3)) heM.RBC2->Fill(_ib->time());
	      if(trig.HasTriggered(4)) heM.RBCTB->Fill(_ib->time());
	      if(trig.HasTriggered(5)) heM.NA->Fill(_ib->time());
	    }//endcap
	  } //iphi
	} //occthresh
      }//loop over hits
      hbP.Etot->Fill(hbPtot);
      hbM.Etot->Fill(hbMtot);
      heP.Etot->Fill(hePtot);
      heM.Etot->Fill(heMtot);
    }//size
  

    if(hoHits.size()>0){
      for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) { // loop over all hits   
	if(_io->energy()>occThresh_){
	  if(_io->id().iphi()<30){
	    if(_io->id().ieta()>10){
	      hoP1tot += _io->energy();
	      hoP1.OCC->Fill(_io->id().ieta(),_io->id().iphi());
	      hoP1.E->Fill(_io->energy());
	      if(trig.HasTriggered(0)) hoP1.DT->Fill(_io->time());
	      if(trig.HasTriggered(1)) hoP1.CSC->Fill(_io->time());
	      if(trig.HasTriggered(2)) hoP1.RBC1->Fill(_io->time());
	      if(trig.HasTriggered(3)) hoP1.RBC2->Fill(_io->time());
	      if(trig.HasTriggered(4)) hoP1.RBCTB->Fill(_io->time());
	      if(trig.HasTriggered(5)) hoP1.NA->Fill(_io->time());
	    }//ieta
	    else{
	      hoP2tot += _io->energy();
	      hoP2.OCC->Fill(_io->id().ieta(),_io->id().iphi());
	      hoP2.E->Fill(_io->energy());
	      if(trig.HasTriggered(0)) hoP2.DT->Fill(_io->time());
	      if(trig.HasTriggered(1)) hoP2.CSC->Fill(_io->time());
	      if(trig.HasTriggered(2)) hoP2.RBC1->Fill(_io->time());
	      if(trig.HasTriggered(3)) hoP2.RBC2->Fill(_io->time());
	      if(trig.HasTriggered(4)) hoP2.RBCTB->Fill(_io->time());
	      if(trig.HasTriggered(5)) hoP2.NA->Fill(_io->time());
	    }//ieta
	  }//iphi
	  else{	    
	    if(_io->id().ieta()>10){
	      hoM1tot += _io->energy();
	      hoM1.OCC->Fill(_io->id().ieta(),_io->id().iphi());
	      hoM1.E->Fill(_io->energy());
	      if(trig.HasTriggered(0)) hoM1.DT->Fill(_io->time());
	      if(trig.HasTriggered(1)) hoM1.CSC->Fill(_io->time());
	      if(trig.HasTriggered(2)) hoM1.RBC1->Fill(_io->time());
	      if(trig.HasTriggered(3)) hoM1.RBC2->Fill(_io->time());
	      if(trig.HasTriggered(4)) hoM1.RBCTB->Fill(_io->time());
	      if(trig.HasTriggered(5)) hoM1.NA->Fill(_io->time());
	    }//ieta
	    else{
	      hoM2tot += _io->energy();
	      hoM2.OCC->Fill(_io->id().ieta(),_io->id().iphi());
	      hoM2.E->Fill(_io->energy());
	      if(trig.HasTriggered(0)) hoM2.DT->Fill(_io->time());
	      if(trig.HasTriggered(1)) hoM2.CSC->Fill(_io->time());
	      if(trig.HasTriggered(2)) hoM2.RBC1->Fill(_io->time());
	      if(trig.HasTriggered(3)) hoM2.RBC2->Fill(_io->time());
	      if(trig.HasTriggered(4)) hoM2.RBCTB->Fill(_io->time());
	      if(trig.HasTriggered(5)) hoM2.NA->Fill(_io->time());
	    }//ieta
	  }//iphi
	}//thresh
      }//loop over hits
      hoP1.Etot->Fill(hoP1tot);
      hoP2.Etot->Fill(hoP2tot);
      hoM1.Etot->Fill(hoM1tot);
      hoM2.Etot->Fill(hoM2tot);
    }//size    
  }//m_dbe

  return;
}


//HB (here i define HB as depth=1 and eta <=15)
//HE (here i define HE as depth=2 and eta>16)
//HO(YB+2) (here i define HO/yb+2 as depth=4 and eta>10
//HO(yb+1,yb0) (depth=4 and eta<=10)
