#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"

HcalRecHitMonitor::HcalRecHitMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1.0;
  ievt_=0;
}

HcalRecHitMonitor::~HcalRecHitMonitor() {
}

void HcalRecHitMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HF");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HO");
    m_dbe->removeContents();

    hbHists.meOCC_MAP_GEO = 0;
    hbHists.meRECHIT_E_all = 0;
    hbHists.meRECHIT_E_low = 0;
    hbHists.meRECHIT_E_tot = 0;
    hbHists.meRECHIT_T_tot = 0;
    hbHists.meRECHIT_E.clear();
    hbHists.meRECHIT_T.clear();

    heHists.meOCC_MAP_GEO = 0;
    heHists.meRECHIT_E_all = 0;
    heHists.meRECHIT_E_low = 0;
    heHists.meRECHIT_E_tot = 0;
    heHists.meRECHIT_T_tot = 0;
    heHists.meRECHIT_E.clear();
    heHists.meRECHIT_T.clear();

    hfHists.meOCC_MAP_GEO = 0;
    hfHists.meRECHIT_E_all = 0;
    hfHists.meRECHIT_E_low = 0;
    hfHists.meRECHIT_E_tot = 0;
    hfHists.meRECHIT_T_tot = 0;
    hfHists.meRECHIT_E.clear();
    hfHists.meRECHIT_T.clear();

    hoHists.meOCC_MAP_GEO = 0;
    hoHists.meRECHIT_E_all = 0;
    hoHists.meRECHIT_E_low = 0;
    hoHists.meRECHIT_E_tot = 0;
    hoHists.meRECHIT_T_tot = 0;
    hoHists.meRECHIT_E.clear();
    hoHists.meRECHIT_T.clear();

    meOCC_MAP_L1=0;
    meOCC_MAP_L1_E=0;
    meOCC_MAP_L2=0;
    meOCC_MAP_L2_E=0;
    meOCC_MAP_L3=0;
    meOCC_MAP_L3_E=0;
    meOCC_MAP_L4=0;
    meOCC_MAP_L4_E=0;
    meRECHIT_E_all= 0;
    meEVT_= 0;

  }

}


namespace HcalRecHitPerChan{
  template<class RecHit>
  inline void perChanHists(int id, const RecHit& rhit, std::map<HcalDetId, MonitorElement*> &toolE, std::map<HcalDetId, MonitorElement*> &toolT, DaqMonitorBEInterface* dbe) {
    
    std::map<HcalDetId,MonitorElement*>::iterator _mei;
    string type = "HB";
    if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HB");
    if(id==1) { 
      type = "HE"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HE");
    }
    else if(id==2) { 
      type = "HO"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HO");
    }
    else if(id==3) { 
      type = "HF"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HF");
    }
    
    ///energies by channel
    _mei=toolE.find(rhit.id()); // look for a histogram with this hit's id
    if (_mei!=toolE.end()){
      if (_mei->second==0) cout << "HcalRecHitMonitor::perChanHists, Found the histo, but it's null??";
      else _mei->second->Fill(rhit.energy()); // if it's there, fill it with energy
    }
    else{
      if(dbe){
	char name[1024];
	sprintf(name,"%s RecHit Energy ieta=%d iphi=%d depth=%d",type.c_str(),rhit.id().ieta(),rhit.id().iphi(),rhit.id().depth());
	toolE[rhit.id()] =  dbe->book1D(name,name,200,0,200); 
	toolE[rhit.id()]->Fill(rhit.energy());
      }
    }
    
    ///times by channel
    _mei=toolT.find(rhit.id()); // look for a histogram with this hit's id
    if (_mei!=toolT.end()){
      if (_mei->second==0) cout << "HcalRecHitMonitor::perChanHists, Found the histo, but it's null??";
      else _mei->second->Fill(rhit.time()); // if it's there, fill it with energy
    }
    else{
      if(dbe){
	char name[1024];
	sprintf(name,"%s RecHit Time ieta=%d iphi=%d depth=%d",type.c_str(),rhit.id().ieta(),rhit.id().iphi(),rhit.id().depth());
	toolT[rhit.id()] =  dbe->book1D(name,name,300,-100,200); 
	toolT[rhit.id()]->Fill(rhit.time());
      }
    }
  }
}

void HcalRecHitMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ){
    doPerChannel_ = true;
  }
  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 41.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -41.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "RecHit eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "RecHit phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;
  
  occThresh_ = ps.getUntrackedParameter<double>("RecHitOccThresh", 1.0);
  cout << "RecHit occupancy threshold set to " << occThresh_ << endl;
  
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor");
    meRECHIT_E_all =  m_dbe->book1D("RecHit Total Energy","RecHit Total Energy",100,0,400);

    meEVT_ = m_dbe->bookInt("RecHit Event Number");    
    meEVT_->Fill(ievt_);
    meOCC_MAP_L1 = m_dbe->book2D("RecHit Depth 1 Occupancy Map","RecHit Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L1_E = m_dbe->book2D("RecHit Depth 1 Energy Map","RecHit Depth 1 Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    meOCC_MAP_L2 = m_dbe->book2D("RecHit Depth 2 Occupancy Map","RecHit Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L2_E = m_dbe->book2D("RecHit Depth 2 Energy Map","RecHit Depth 2 Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    meOCC_MAP_L3 = m_dbe->book2D("RecHit Depth 3 Occupancy Map","RecHit Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L3_E = m_dbe->book2D("RecHit Depth 3 Energy Map","RecHit Depth 3 Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    meOCC_MAP_L4 = m_dbe->book2D("RecHit Depth 4 Occupancy Map","RecHit Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    meOCC_MAP_L4_E = m_dbe->book2D("RecHit Depth 4 Energy Map","RecHit Depth 4 Energy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    
    meOCC_MAP_ETA = m_dbe->book1D("RecHit Eta Occupancy Map","RecHit Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    meOCC_MAP_PHI = m_dbe->book1D("RecHit Phi Occupancy Map","RecHit Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    meOCC_MAP_ETA_E = m_dbe->book1D("RecHit Eta Energy Map","RecHit Eta Energy Map",etaBins_,etaMin_,etaMax_);
    meOCC_MAP_PHI_E = m_dbe->book1D("RecHit Phi Energy Map","RecHit Phi Energy Map",phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HB");
    hbHists.meRECHIT_E_tot = m_dbe->book1D("HB RecHit Total Energy","HB RecHit Total Energy",100,0,400);
    hbHists.meRECHIT_E_all = m_dbe->book1D("HB RecHit Energies","HB RecHit Energies",200,0,200);
    hbHists.meRECHIT_E_low = m_dbe->book1D("HB RecHit Energies - Low Region","HB RecHit Energies - Low Region",200,0,10);
    hbHists.meRECHIT_T_tot = m_dbe->book1D("HB RecHit Times","HB RecHit Times",300,-100,200);
    hbHists.meOCC_MAP_GEO = m_dbe->book2D("HB RecHit Geo Occupancy Map","HB RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HE");
    heHists.meRECHIT_E_tot = m_dbe->book1D("HE RecHit Total Energy","HE RecHit Total Energy",100,0,400);
    heHists.meRECHIT_E_all = m_dbe->book1D("HE RecHit Energies","HE RecHit Energies",200,0,200);
    heHists.meRECHIT_E_low = m_dbe->book1D("HE RecHit Energies - Low Region","HE RecHit Energies - Low Region",200,0,10);
    heHists.meRECHIT_T_tot = m_dbe->book1D("HE RecHit Times","HE RecHit Times",300,-100,200);
    heHists.meOCC_MAP_GEO = m_dbe->book2D("HE RecHit Geo Occupancy Map","HE RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HF");
    hfHists.meRECHIT_E_tot = m_dbe->book1D("HF RecHit Total Energy","HF RecHit Total Energy",100,0,400);
    hfHists.meRECHIT_E_all = m_dbe->book1D("HF RecHit Energies","HF RecHit Energies",200,0,200);
    hfHists.meRECHIT_E_low = m_dbe->book1D("HF RecHit Energies - Low Region","HF RecHit Energies - Low Region",200,0,10);
hfHists.meRECHIT_T_tot = m_dbe->book1D("HF RecHit Times","HF RecHit Times",300,-100,200);
    hfHists.meOCC_MAP_GEO = m_dbe->book2D("HF RecHit Geo Occupancy Map","HF RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HO");
    hoHists.meRECHIT_E_tot = m_dbe->book1D("HO RecHit Total Energy","HO RecHit Total Energy",100,0,400);
    hoHists.meRECHIT_E_all = m_dbe->book1D("HO RecHit Energies","HO RecHit Energies",200,0,200);
    hoHists.meRECHIT_E_low = m_dbe->book1D("HO RecHit Energies - Low Region","HO RecHit Energies - Low Region",200,0,10);
hoHists.meRECHIT_T_tot = m_dbe->book1D("HO RecHit Times","HO RecHit Times",300,-100,200);
    hoHists.meOCC_MAP_GEO = m_dbe->book2D("HO RecHit Geo Occupancy Map","HO RecHit Geo Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
  }

  return;
}

void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				     const HORecHitCollection& hoHits, 
				     const HFRecHitCollection& hfHits){

  if(!m_dbe) { printf("HcalRecHitMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  ievt_++;
  meEVT_->Fill(ievt_);

  HBHERecHitCollection::const_iterator _ib;
  HORecHitCollection::const_iterator _io;
  HFRecHitCollection::const_iterator _if;
  float tot = 0, tot2=0, all =0;
  
  if(hbHits.size()>0){    
    for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits
      const HBHERecHit rec = (const HBHERecHit)(*_ib);

      if(rec.energy()>0.0){
	if((HcalSubdetector)(rec.id().subdet())==HcalBarrel){
	  hbHists.meRECHIT_E_all->Fill(rec.energy());
	  hbHists.meRECHIT_E_low->Fill(rec.energy());
	  hbHists.meRECHIT_T_tot->Fill(rec.time());
	  
	  tot += rec.energy();
	  if(rec.energy()>occThresh_){
	    hbHists.meOCC_MAP_GEO->Fill(rec.id().ieta(),rec.id().iphi());
	    meOCC_MAP_ETA->Fill(rec.id().ieta());
	    meOCC_MAP_PHI->Fill(rec.id().iphi());
	    
	    meOCC_MAP_ETA_E->Fill(rec.id().ieta(),rec.energy());
	    meOCC_MAP_PHI_E->Fill(rec.id().iphi(),rec.energy());
	    
	    if(rec.id().depth()==1){ 
	      meOCC_MAP_L1->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L1_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	    else if(rec.id().depth()==2){ 
	      meOCC_MAP_L2->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L2_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	    else if(rec.id().depth()==3){ 
	      meOCC_MAP_L3->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L3_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	    if(rec.id().depth()==4){ 
	      meOCC_MAP_L4->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L4_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	  }      
	  if(doPerChannel_) HcalRecHitPerChan::perChanHists<HBHERecHit>(0,*_ib,hbHists.meRECHIT_E,hbHists.meRECHIT_T,m_dbe);
	}
	else if((HcalSubdetector)(rec.id().subdet())==HcalEndcap){
	  heHists.meRECHIT_E_all->Fill(rec.energy());
	  heHists.meRECHIT_E_low->Fill(rec.energy());
	  heHists.meRECHIT_T_tot->Fill(rec.time());

	  tot2 += rec.energy();
	  if(rec.energy()>occThresh_){
	    meOCC_MAP_ETA->Fill(rec.id().ieta());
	    meOCC_MAP_PHI->Fill(rec.id().iphi());
	    meOCC_MAP_ETA_E->Fill(rec.id().ieta(),rec.energy());
	    meOCC_MAP_PHI_E->Fill(rec.id().iphi(),rec.energy());


	    heHists.meOCC_MAP_GEO->Fill(rec.id().ieta(),rec.id().iphi());
	    if(rec.id().depth()==1){ 
	      meOCC_MAP_L1->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L1_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	    else if(rec.id().depth()==2){ 
	      meOCC_MAP_L2->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L2_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	    else if(rec.id().depth()==3){ 
	      meOCC_MAP_L3->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L3_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	    if(rec.id().depth()==4){ 
	      meOCC_MAP_L4->Fill(rec.id().ieta(),rec.id().iphi());
	      meOCC_MAP_L4_E->Fill(rec.id().ieta(),rec.id().iphi(), rec.energy());
	    }
	  }      
	  if(doPerChannel_) HcalRecHitPerChan::perChanHists<HBHERecHit>(1,*_ib,heHists.meRECHIT_E,heHists.meRECHIT_T,m_dbe);
	}
      }
      
    }
    if(tot>0) hbHists.meRECHIT_E_tot->Fill(tot);
    if(tot2>0) heHists.meRECHIT_E_tot->Fill(tot2);
    all += tot;
    all += tot2;
  }
  
  tot = 0;
  if(hoHits.size()>0){
    for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) { // loop over all hits
    
      if(_io->energy()>0.0){
	hoHists.meRECHIT_E_all->Fill(_io->energy());
	hoHists.meRECHIT_E_low->Fill(_io->energy());
	hoHists.meRECHIT_T_tot->Fill(_io->time());

	tot += _io->energy();
	if(_io->energy()>occThresh_){
	  meOCC_MAP_ETA->Fill(_io->id().ieta());
	  meOCC_MAP_PHI->Fill(_io->id().iphi());
	  meOCC_MAP_ETA_E->Fill(_io->id().ieta(),_io->energy());
	  meOCC_MAP_PHI_E->Fill(_io->id().iphi(),_io->energy());
	  
	  hoHists.meOCC_MAP_GEO->Fill(_io->id().ieta(),_io->id().iphi());
	  if(_io->id().depth()==1){ 
	    meOCC_MAP_L1->Fill(_io->id().ieta(),_io->id().iphi());
	    meOCC_MAP_L1_E->Fill(_io->id().ieta(),_io->id().iphi(), _io->energy());
	  }
	  else if(_io->id().depth()==2){ 
	    meOCC_MAP_L2->Fill(_io->id().ieta(),_io->id().iphi());
	    meOCC_MAP_L2_E->Fill(_io->id().ieta(),_io->id().iphi(), _io->energy());
	  }
	  else if(_io->id().depth()==3){ 
	    meOCC_MAP_L3->Fill(_io->id().ieta(),_io->id().iphi());
	    meOCC_MAP_L3_E->Fill(_io->id().ieta(),_io->id().iphi(), _io->energy());
	  }
	  if(_io->id().depth()==4){ 
	    meOCC_MAP_L4->Fill(_io->id().ieta(),_io->id().iphi());
	    meOCC_MAP_L4_E->Fill(_io->id().ieta(),_io->id().iphi(), _io->energy());
	  }
	}
	if(doPerChannel_) HcalRecHitPerChan::perChanHists<HORecHit>(2,*_io,hoHists.meRECHIT_E,hoHists.meRECHIT_T,m_dbe);
      }
    }
    if(tot>0) hoHists.meRECHIT_E_tot->Fill(tot);
    all += tot;
  }
  
  tot=0;
  if(hfHits.size()>0){
    for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) { // loop over all hits
      if(_if->energy()>0.0){
	hfHists.meRECHIT_E_all->Fill(_if->energy());
	hfHists.meRECHIT_E_low->Fill(_if->energy());
	hfHists.meRECHIT_T_tot->Fill(_if->time());

	tot += _if->energy();
	if(_if->energy()>occThresh_){
	  meOCC_MAP_ETA->Fill(_if->id().ieta());
	  meOCC_MAP_PHI->Fill(_if->id().iphi());
	  meOCC_MAP_ETA_E->Fill(_if->id().ieta(),_if->energy());
	  meOCC_MAP_PHI_E->Fill(_if->id().iphi(),_if->energy());
	  
	  hfHists.meOCC_MAP_GEO->Fill(_if->id().ieta(),_if->id().iphi());
	  if(_if->id().depth()==1){ 
	    meOCC_MAP_L1->Fill(_if->id().ieta(),_if->id().iphi());
	    meOCC_MAP_L1_E->Fill(_if->id().ieta(),_if->id().iphi(), _if->energy());
	  }
	  else if(_if->id().depth()==2){ 
	    meOCC_MAP_L2->Fill(_if->id().ieta(),_if->id().iphi());
	    meOCC_MAP_L2_E->Fill(_if->id().ieta(),_if->id().iphi(), _if->energy());
	  }
	  else if(_if->id().depth()==3){ 
	    meOCC_MAP_L3->Fill(_if->id().ieta(),_if->id().iphi());
	    meOCC_MAP_L3_E->Fill(_if->id().ieta(),_if->id().iphi(), _if->energy());
	  }
	  if(_if->id().depth()==4){ 
	    meOCC_MAP_L4->Fill(_if->id().ieta(),_if->id().iphi());
	    meOCC_MAP_L4_E->Fill(_if->id().ieta(),_if->id().iphi(), _if->energy());
	  }
	}
	if(doPerChannel_) HcalRecHitPerChan::perChanHists<HFRecHit>(3,*_if,hfHists.meRECHIT_E,hfHists.meRECHIT_T,m_dbe);
      }
    }
    if(tot>0) hfHists.meRECHIT_E_tot->Fill(tot);
    all += tot;
  }
  
  if(all>0) meRECHIT_E_all->Fill(all);
  return;
}

