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
    

    hbHists.meOCC_MAP_GEO = 0;
    hbHists.meEN_MAP_GEO = 0;
    hbHists.meMAX_E = 0;
    hbHists.meMAX_T = 0;

    heHists.meOCC_MAP_GEO = 0;
    heHists.meEN_MAP_GEO = 0;
    heHists.meMAX_E = 0;
    heHists.meMAX_T = 0;

    hfHists.meOCC_MAP_GEO = 0;
    hfHists.meEN_MAP_GEO = 0;
    hfHists.meMAX_E = 0;
    hfHists.meMAX_T = 0;

    hoHists.meOCC_MAP_GEO = 0;
    hoHists.meEN_MAP_GEO = 0;
    hoHists.meMAX_E = 0;
    hoHists.meMAX_T = 0;

    meOCC_MAP_all_GEO= 0;
    meEN_MAP_all_GEO= 0;
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

  occThresh_ = ps.getUntrackedParameter<double>("HotCellThresh", 0);
  cout << "Hot Cell threshold set to " << occThresh_ << endl;

  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor");

    meEVT_ = m_dbe->bookInt("HotCell Task Event Number");    
    meEVT_->Fill(ievt_);

    meMAX_E_all =  m_dbe->book1D("HotCell Energy","HotCell Energy",100,0,400);
    meMAX_T_all =  m_dbe->book1D("HotCell Time","HotCell Time",100,0,200);
    meOCC_MAP_all_GEO  = m_dbe->book2D("HotCell Geo Occupancy Map","HotCell Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);


    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HB");
    hbHists.meMAX_E =  m_dbe->book1D("HB HotCell Energy","HB HotCell Energy",100,0,400);
    hbHists.meMAX_T =  m_dbe->book1D("HB HotCell Time","HB HotCell Time",100,-100,200);
    hbHists.meMAX_ID =  m_dbe->book1D("HB HotCell ID","HB HotCell ID",10000,1000,12000);
    hbHists.meOCC_MAP_GEO  = m_dbe->book2D("HB HotCell Geo Occupancy Map","HB HotCell Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.meEN_MAP_GEO  = m_dbe->book2D("HB HotCell Geo Energy Map","HB HotCell Geo Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HE");
    heHists.meMAX_E =  m_dbe->book1D("HE HotCell Energy","HE HotCell Energy",100,0,400);
    heHists.meMAX_T =  m_dbe->book1D("HE HotCell Time","HE HotCell Time",100,-100,200);
    heHists.meMAX_ID =  m_dbe->book1D("HE HotCell ID","HE HotCell ID",4000,1000,5000);
    heHists.meOCC_MAP_GEO  = m_dbe->book2D("HE HotCell Geo Occupancy Map","HE HotCell Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.meEN_MAP_GEO  = m_dbe->book2D("HE HotCell Geo Energy Map","HE HotCell Geo Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HF");
    hfHists.meMAX_E =  m_dbe->book1D("HF HotCell Energy","HF HotCell Energy",100,0,400);
    hfHists.meMAX_T =  m_dbe->book1D("HF HotCell Time","HF HotCell Time",100,-100,200);
    hfHists.meMAX_ID =  m_dbe->book1D("HF HotCell ID","HF HotCell ID",10000,0,10000);
    hfHists.meOCC_MAP_GEO  = m_dbe->book2D("HF HotCell Geo Occupancy Map","HF HotCell Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.meEN_MAP_GEO  = m_dbe->book2D("HF HotCell Geo Energy Map","HF HotCell Geo Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/HotCellMonitor/HO");
    hoHists.meMAX_E =  m_dbe->book1D("HO HotCell Energy","HO HotCell Energy",100,0,400);
    hoHists.meMAX_T =  m_dbe->book1D("HO HotCell Time","HO HotCell Time",100,-100,200);
    hoHists.meMAX_ID =  m_dbe->book1D("HO HotCell ID","HO HotCell ID",1000,4000,5000);
    hoHists.meOCC_MAP_GEO  = m_dbe->book2D("HO HotCell Geo Occupancy Map","HO HotCell Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.meEN_MAP_GEO  = m_dbe->book2D("HO HotCell Geo Energy Map","HO HotCell Geo Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
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
  
  if(hbHits.size()>0){
    for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits      
      //      if(_ib->id().ieta()==9 && _ib->id().iphi()==17) printf("rawid: %d\n",_ib->id().rawId());
      if((HcalSubdetector)(_ib->id().subdet())!=HcalBarrel) continue;
      if(_ib->energy()>occThresh_){	
	if(vetoCell(_ib->id())) continue;
	hbHists.meEN_MAP_GEO->Fill(_ib->id().ieta(),_ib->id().iphi(),_ib->energy());
	if(_ib->energy()>enS){
	  enS = _ib->energy();
	  tS = _ib->time();
	  etaS = _ib->id().ieta();
	  phiS = _ib->id().iphi();
	  idS = etaS+100*phiS+1000*_ib->id().depth();
	}
      }
    }
    if(enS>0){
      hbHists.meMAX_E->Fill(enS);
      hbHists.meMAX_T->Fill(tS);
      hbHists.meOCC_MAP_GEO->Fill(etaS,phiS);
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
      if(_ib->energy()>occThresh_){	
	if(vetoCell(_ib->id())) continue;
	heHists.meEN_MAP_GEO->Fill(_ib->id().ieta(),_ib->id().iphi(),_ib->energy());
	if(_ib->energy()>enS){
	  enS = _ib->energy();
	  tS = _ib->time();
	  etaS = _ib->id().ieta();
	  phiS = _ib->id().iphi();
	  idS = etaS+100*phiS+1000*_ib->id().depth();
	}
      }
    }
    if(enS>0){
      heHists.meMAX_E->Fill(enS);
      heHists.meMAX_T->Fill(tS);
      heHists.meOCC_MAP_GEO->Fill(etaS,phiS);
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
      if(_io->energy()>occThresh_){
	if(vetoCell(_io->id())) continue;
	hoHists.meEN_MAP_GEO->Fill(_io->id().ieta(),_io->id().iphi(),_io->energy());
	if(_io->energy()>enS){
	  enS = _io->energy();
	  tS = _io->time();
	  etaS = _io->id().ieta();
	  phiS = _io->id().iphi();
	  idS = etaS+100*phiS+1000*_io->id().depth();
	}
      }
    }
    if(enS>0){
      hoHists.meMAX_E->Fill(enS);
      hoHists.meMAX_T->Fill(tS);
      hoHists.meOCC_MAP_GEO->Fill(etaS,phiS);
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
      if(_if->energy()>occThresh_){
	if(vetoCell(_if->id())) continue;
	hfHists.meEN_MAP_GEO->Fill(_if->id().ieta(),_if->id().iphi(),_if->energy());
	if(_if->energy()>enS){
	  enS = _if->energy();
	  tS = _if->time();
	  etaS = _if->id().ieta();
	  phiS = _if->id().iphi();
	  idS = etaS+100*phiS+1000*_if->id().depth();
	}
      }
    }
    if(enS>0){
      hfHists.meMAX_E->Fill(enS);
      hfHists.meMAX_T->Fill(tS);
      hfHists.meOCC_MAP_GEO->Fill(etaS,phiS);
      hfHists.meMAX_ID->Fill(idS);
    }
  }
  if(enS>enA){
    enA = enS;
    tA = tS;
    etaA = etaS;
    phiA = phiS;
  }

 if(enA>0){
   meMAX_E_all->Fill(enA);
   meMAX_T_all->Fill(tA);
   meOCC_MAP_all_GEO->Fill(etaA,phiA);
 }

  return;
}

