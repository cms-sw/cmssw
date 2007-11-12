#include "DQM/HcalMonitorTasks/interface/HcalHotCellMonitor.h"

HcalHotCellMonitor::HcalHotCellMonitor() {
  ievt_=0;
}

HcalHotCellMonitor::~HcalHotCellMonitor() {
}

void HcalHotCellMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HF");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HO");
    m_dbe->removeContents();
    

    hbHists.meOCC_MAP_GEO_Thr0 = 0;
    hbHists.meEN_MAP_GEO_Thr0 = 0;
    hbHists.meOCC_MAP_GEO_Thr1 = 0;
    hbHists.meEN_MAP_GEO_Thr1 = 0;
    hbHists.meOCC_MAP_GEO_Max = 0;
    hbHists.meEN_MAP_GEO_Max = 0;
    hbHists.meMAX_E = 0;
    hbHists.meMAX_T = 0;

    heHists.meOCC_MAP_GEO_Thr0 = 0;
    heHists.meEN_MAP_GEO_Thr0 = 0;
    heHists.meOCC_MAP_GEO_Thr1 = 0;
    heHists.meEN_MAP_GEO_Thr1 = 0;
    heHists.meOCC_MAP_GEO_Max = 0;
    heHists.meEN_MAP_GEO_Max = 0;
    heHists.meMAX_E = 0;
    heHists.meMAX_T = 0;

    hfHists.meOCC_MAP_GEO_Thr0 = 0;
    hfHists.meEN_MAP_GEO_Thr0 = 0;
    hfHists.meOCC_MAP_GEO_Thr1 = 0;
    hfHists.meEN_MAP_GEO_Thr1 = 0;
    hfHists.meOCC_MAP_GEO_Max = 0;
    hfHists.meEN_MAP_GEO_Max = 0;
    hfHists.meMAX_E = 0;
    hfHists.meMAX_T = 0;

    hoHists.meOCC_MAP_GEO_Thr0 = 0;
    hoHists.meEN_MAP_GEO_Thr0 = 0;
    hoHists.meOCC_MAP_GEO_Thr1 = 0;
    hoHists.meEN_MAP_GEO_Thr1 = 0;
    hoHists.meOCC_MAP_GEO_Max = 0;
    hoHists.meEN_MAP_GEO_Max = 0;
    hoHists.meMAX_E = 0;
    hoHists.meMAX_T = 0;

    meMAX_E_all= 0;
    meMAX_T_all= 0;
    meEVT_= 0;

  }

}


void HcalHotCellMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "HotCell eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "HotCell phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;

  occThresh0_ = ps.getUntrackedParameter<double>("HotCellThresh0", 0);
  occThresh1_ = ps.getUntrackedParameter<double>("HotCellThresh1", 5);
  cout << "Hot Cell thresholds set to " << occThresh0_ << " "<< occThresh1_ <<endl;

  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor");

    meEVT_ = m_dbe->bookInt("HotCell Task Event Number");    
    meEVT_->Fill(ievt_);

    meMAX_E_all =  m_dbe->book1D("HotCell Energy","HotCell Energy",200,0,1000);
    meMAX_T_all =  m_dbe->book1D("HotCell Time","HotCell Time",200,-50,300);
    
    meOCC_MAP_L1= m_dbe->book2D("HotCell Depth 1 Occupancy Map","HotCell Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L1= m_dbe->book2D("HotCell Depth 1 Energy Map","HotCell Depth 1 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L2= m_dbe->book2D("HotCell Depth 2 Occupancy Map","HotCell Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L2= m_dbe->book2D("HotCell Depth 2 Energy Map","HotCell Depth 2 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L3= m_dbe->book2D("HotCell Depth 3 Occupancy Map","HotCell Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L3= m_dbe->book2D("HotCell Depth 3 Energy Map","HotCell Depth 3 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_L4= m_dbe->book2D("HotCell Depth 4 Occupancy Map","HotCell Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_L4= m_dbe->book2D("HotCell Depth 4 Energy Map","HotCell Depth 4 Energy Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_all = m_dbe->book2D("HotCell Occupancy Map","HotCell Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meEN_MAP_all  = m_dbe->book2D("HotCell Energy Map","HotCell Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HB");
    hbHists.meMAX_E =  m_dbe->book1D("HB HotCell Energy","HB HotCell Energy",200,0,1000);
    hbHists.meMAX_T =  m_dbe->book1D("HB HotCell Time","HB HotCell Time",200,-50,300);
    hbHists.meMAX_ID =  m_dbe->book1D("HB HotCell ID","HB HotCell ID",10000,1000,12000);
    
    hbHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HB HotCell Geo Occupancy Map, Threshold 0","HB HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HB HotCell Geo Energy Map, Threshold 0","HB HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HB HotCell Geo Occupancy Map, Threshold 1","HB HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HB HotCell Geo Energy Map, Threshold 1","HB HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HB HotCell Geo Occupancy Map, Max Cell","HB HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HB HotCell Geo Energy Map, Max Cell","HB HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HE");
    heHists.meMAX_E =  m_dbe->book1D("HE HotCell Energy","HE HotCell Energy",200,0,1000);
    heHists.meMAX_T =  m_dbe->book1D("HE HotCell Time","HE HotCell Time",200,-50,300);
    heHists.meMAX_ID =  m_dbe->book1D("HE HotCell ID","HE HotCell ID",4000,1000,5000);
    heHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HE HotCell Geo Occupancy Map, Threshold 0","HE HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HE HotCell Geo Energy Map, Threshold 0","HE HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HE HotCell Geo Occupancy Map, Threshold 1","HE HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HE HotCell Geo Energy Map, Threshold 1","HE HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HE HotCell Geo Occupancy Map, Max Cell","HE HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HE HotCell Geo Energy Map, Max Cell","HE HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HF");
    hfHists.meMAX_E =  m_dbe->book1D("HF HotCell Energy","HF HotCell Energy",200,0,1000);
    hfHists.meMAX_T =  m_dbe->book1D("HF HotCell Time","HF HotCell Time",200,-50,300);
    hfHists.meMAX_ID =  m_dbe->book1D("HF HotCell ID","HF HotCell ID",10000,0,10000);
    hfHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HF HotCell Geo Occupancy Map, Threshold 0","HF HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HF HotCell Geo Energy Map, Threshold 0","HF HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HF HotCell Geo Occupancy Map, Threshold 1","HF HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HF HotCell Geo Energy Map, Threshold 1","HF HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HF HotCell Geo Occupancy Map, Max Cell","HF HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HF HotCell Geo Energy Map, Max Cell","HF HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HO");
    hoHists.meMAX_E =  m_dbe->book1D("HO HotCell Energy","HO HotCell Energy",200,0,1000);
    hoHists.meMAX_T =  m_dbe->book1D("HO HotCell Time","HO HotCell Time",200,-50,300);
    hoHists.meMAX_ID =  m_dbe->book1D("HO HotCell ID","HO HotCell ID",1000,4000,5000);
    hoHists.meOCC_MAP_GEO_Thr0  = m_dbe->book2D("HO HotCell Geo Occupancy Map, Threshold 0","HO HotCell Geo Occupancy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Thr0  = m_dbe->book2D("HO HotCell Geo Energy Map, Threshold 0","HO HotCell Geo Energy Map, Threshold 0",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meOCC_MAP_GEO_Thr1  = m_dbe->book2D("HO HotCell Geo Occupancy Map, Threshold 1","HO HotCell Geo Occupancy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Thr1  = m_dbe->book2D("HO HotCell Geo Energy Map, Threshold 1","HO HotCell Geo Energy Map, Threshold 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meOCC_MAP_GEO_Max  = m_dbe->book2D("HO HotCell Geo Occupancy Map, Max Cell","HO HotCell Geo Occupancy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO_Max  = m_dbe->book2D("HO HotCell Geo Energy Map, Max Cell","HO HotCell Geo Energy Map, Max Cell",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  }

  return;
}

void HcalHotCellMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  if(!m_dbe) { printf("HcalHotCellMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  ievt_++;
  meEVT_->Fill(ievt_);

  HBHERecHitCollection::const_iterator _ib;
  HORecHitCollection::const_iterator _io;
  HFRecHitCollection::const_iterator _if;

  float enS=0, tS=0, etaS=0, phiS=0, idS=-1;
  float enA=0, tA=0, etaA=0, phiA=0;
  int depth=0;
  
  if(hbHits.size()>0){
    for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits      
      //      if(_ib->id().ieta()==9 && _ib->id().iphi()==17) printf("rawid: %d\n",_ib->id().rawId());
      if((HcalSubdetector)(_ib->id().subdet())!=HcalBarrel) continue;
      if(_ib->energy()>occThresh0_){	
	if(vetoCell(_ib->id())) continue;
	hbHists.meEN_MAP_GEO_Thr0->Fill(_ib->id().ieta(),_ib->id().iphi(),_ib->energy());
	hbHists.meOCC_MAP_GEO_Thr0->Fill(_ib->id().ieta(),_ib->id().iphi());
	if(_ib->energy()>occThresh1_){
	  hbHists.meEN_MAP_GEO_Thr1->Fill(_ib->id().ieta(),_ib->id().iphi(),_ib->energy());
	  hbHists.meOCC_MAP_GEO_Thr1->Fill(_ib->id().ieta(),_ib->id().iphi());
	}
	if(_ib->energy()>enS){
	  enS = _ib->energy();
	  tS = _ib->time();
	  etaS = _ib->id().ieta();
	  phiS = _ib->id().iphi();
	  idS = 1000*etaS;
	  if(idS<0) idS -= (10*phiS+depth);
	  else idS += (10*phiS+depth);
	  depth = _ib->id().depth();
	}
      }
    }
    if(enS>occThresh0_){
      hbHists.meMAX_E->Fill(enS);
      hbHists.meMAX_T->Fill(tS);
      hbHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
      hbHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
      hbHists.meMAX_ID->Fill(idS);
    }
    if(enS>enA){
      enA = enS;
      tA = tS;
      etaA = etaS;
      phiA = phiS;
    }

    enS=0, tS=0, etaS=0, phiS=0;
    for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits
      if((HcalSubdetector)(_ib->id().subdet())!=HcalEndcap) continue;
      if(_ib->energy()>occThresh0_){	
	if(vetoCell(_ib->id())) continue;
	heHists.meEN_MAP_GEO_Thr0->Fill(_ib->id().ieta(),_ib->id().iphi(),_ib->energy());
	heHists.meOCC_MAP_GEO_Thr0->Fill(_ib->id().ieta(),_ib->id().iphi());
	if(_ib->energy()>occThresh1_){
	  heHists.meEN_MAP_GEO_Thr1->Fill(_ib->id().ieta(),_ib->id().iphi(),_ib->energy());
	  heHists.meOCC_MAP_GEO_Thr1->Fill(_ib->id().ieta(),_ib->id().iphi());
	}
	if(_ib->energy()>enS){
	  enS = _ib->energy();
	  tS = _ib->time();
	  etaS = _ib->id().ieta();
	  phiS = _ib->id().iphi();
	  idS = 1000*etaS;
	  if(idS<0) idS -= (10*phiS+depth);
	  else idS += (10*phiS+depth);
	  depth = _ib->id().depth();
	}
      }
    }
    if(enS>0){
      heHists.meMAX_E->Fill(enS);
      heHists.meMAX_T->Fill(tS);
      heHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
      heHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
      heHists.meMAX_ID->Fill(idS);
    }
    if(enS>enA){
      enA = enS;
      tA = tS;
      etaA = etaS;
      phiA = phiS;
    }  
  }


  enS=0, tS=0, etaS=0, phiS=0;
  if(hoHits.size()>0){
    for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) { // loop over all hits
      if(_io->energy()>occThresh0_){
	if(vetoCell(_io->id())) continue;
	hoHists.meEN_MAP_GEO_Thr0->Fill(_io->id().ieta(),_io->id().iphi(),_io->energy());
	hoHists.meOCC_MAP_GEO_Thr0->Fill(_io->id().ieta(),_io->id().iphi());
	if(_io->energy()>occThresh1_){
	  hoHists.meEN_MAP_GEO_Thr1->Fill(_io->id().ieta(),_io->id().iphi(),_io->energy());
	  hoHists.meOCC_MAP_GEO_Thr1->Fill(_io->id().ieta(),_io->id().iphi());
	}
	if(_io->energy()>enS){
	  enS = _io->energy();
	  tS = _io->time();
	  etaS = _io->id().ieta();
	  phiS = _io->id().iphi();
	  idS = 1000*etaS;
	  if(idS<0) idS -= (10*phiS+depth);
	  else idS += (10*phiS+depth);
	  depth = _io->id().depth();
	}
      }
    }
    if(enS>0){
      hoHists.meMAX_E->Fill(enS);
      hoHists.meMAX_T->Fill(tS);
      hoHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
      hoHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
      hoHists.meMAX_ID->Fill(idS);
    }
  }
  if(enS>enA){
    enA = enS;
    tA = tS;
    etaA = etaS;
    phiA = phiS;
  }

  enS=0, tS=0, etaS=0, phiS=0;
  if(hfHits.size()>0){
    for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) { // loop over all hits
      if(_if->energy()>occThresh0_){
	if(vetoCell(_if->id())) continue;
	hfHists.meEN_MAP_GEO_Thr0->Fill(_if->id().ieta(),_if->id().iphi(),_if->energy());
	hfHists.meOCC_MAP_GEO_Thr0->Fill(_if->id().ieta(),_if->id().iphi());
	if(_if->energy()>occThresh1_){
	  hfHists.meEN_MAP_GEO_Thr1->Fill(_if->id().ieta(),_if->id().iphi(),_if->energy());
	  hfHists.meOCC_MAP_GEO_Thr1->Fill(_if->id().ieta(),_if->id().iphi());
	}
	if(_if->energy()>enS){
	  enS = _if->energy();
	  tS = _if->time();
	  etaS = _if->id().ieta();
	  phiS = _if->id().iphi();
	  idS = 1000*etaS;
	  if(idS<0) idS -= (10*phiS+depth);
	  else idS += (10*phiS+depth);

	  depth = _if->id().depth();
	}
      }
    }
    if(enS>0){
      hfHists.meMAX_E->Fill(enS);
      hfHists.meMAX_T->Fill(tS);
      hfHists.meOCC_MAP_GEO_Max->Fill(etaS,phiS);
      hfHists.meEN_MAP_GEO_Max->Fill(etaS,phiS,enS);
      hfHists.meMAX_ID->Fill(idS);
    }
  }
  if(enS>enA){
    enA = enS;
    tA = tS;
    etaA = etaS;
    phiA = phiS;
  }

  if(enA>occThresh0_){
    meMAX_E_all->Fill(enA);
    meMAX_T_all->Fill(tA);
    meOCC_MAP_all->Fill(etaA,phiA);
    meEN_MAP_all->Fill(etaA,phiA,enA);
    
    if(depth==1){
      meOCC_MAP_L1->Fill(etaA,phiA);
      meEN_MAP_L1->Fill(etaA,phiA,enA);
    }
    else if(depth==2){
      meOCC_MAP_L2->Fill(etaA,phiA);
      meEN_MAP_L2->Fill(etaA,phiA,enA);
    }
    else if(depth==3){
      meOCC_MAP_L3->Fill(etaA,phiA);
      meEN_MAP_L3->Fill(etaA,phiA,enA);
    }
    else if(depth==4){
      meOCC_MAP_L4->Fill(etaA,phiA);
      meEN_MAP_L4->Fill(etaA,phiA,enA);
    }
  }
  
  return;
}

