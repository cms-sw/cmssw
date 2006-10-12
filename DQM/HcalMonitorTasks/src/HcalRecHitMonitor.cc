#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"

HcalRecHitMonitor::HcalRecHitMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1.0;
  ievt_=0;
}

HcalRecHitMonitor::~HcalRecHitMonitor() {}


namespace HcalRecHitPerChan{
  template<class RecHit>
  inline void perChanHists(int id, const RecHit& rhit, std::map<HcalDetId, MonitorElement*> &toolE, std::map<HcalDetId, MonitorElement*> &toolT, DaqMonitorBEInterface* dbe) {
    
    std::map<HcalDetId,MonitorElement*>::iterator _mei;
    string type = "HBHE";
    if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HBHE");
    if(id==0) { 
      type = "HBHE"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HBHE");
    }
    else if(id==1) { 
      type = "HO"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HO");
    }
    else if(id==2) { 
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
	toolE[rhit.id()] =  dbe->book1D(name,name,100,0,500); 
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
	toolT[rhit.id()] =  dbe->book1D(name,name,100,0,500); 
	toolT[rhit.id()]->Fill(rhit.time());
      }
    }
  }
}


void HcalRecHitMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ) {
    doPerChannel_ = true;
  }
  
  occThresh_ = ps.getUntrackedParameter<double>("RecHitOccThresh", 1.0);
  cout << "RecHit occupancy threshold set to " << occThresh_ << endl;

  ievt_=0;

  if ( m_dbe != NULL) {

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor");
    meRECHIT_E_all =  m_dbe->book1D("RecHit Total Energy","RecHit Total Energy",100,0,2000);
    meOCC_MAP_all_GEO  = m_dbe->book2D("RecHit Geo Occupancy Map","RecHit Geo Occupancy Map",59,-29.5,29.5,74,0,73);

    meEVT_ = m_dbe->bookInt("RecHit Event Number");    
    meEVT_->Fill(ievt_);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HBHE");
    hbHists.meRECHIT_E_tot = m_dbe->book1D("HBHE RecHit Total Energy","HBHE RecHit Total Energy",100,0,2000);
    hbHists.meRECHIT_E_all = m_dbe->book1D("HBHE RecHit Energies","HBHE RecHit Energies",100,0,1000);
    hbHists.meRECHIT_T_tot = m_dbe->book1D("HBHE RecHit Times","HBHE RecHit Times",100,0,500);
    hbHists.meOCC_MAP_GEO = m_dbe->book2D("HBHE RecHit Geo Occupancy Map","RecHit Geo Occupancy Map",59,-29.5,29.5,74,0,73);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HF");
    hfHists.meRECHIT_E_tot = m_dbe->book1D("HF RecHit Total Energy","HF RecHit Total Energy",100,0,2000);
    hfHists.meRECHIT_E_all = m_dbe->book1D("HF RecHit Energies","HF RecHit Energies",100,0,1000);
    hfHists.meRECHIT_T_tot = m_dbe->book1D("HF RecHit Times","HF RecHit Times",100,0,500);
    hfHists.meOCC_MAP_GEO = m_dbe->book2D("HF RecHit Geo Occupancy Map","RecHit Geo Occupancy Map",59,-29.5,29.5,74,0,73);

    m_dbe->setCurrentFolder("HcalMonitor/RecHitMonitor/HO");
    hoHists.meRECHIT_E_tot = m_dbe->book1D("HO RecHit Total Energy","HO RecHit Total Energy",100,0,2000);
    hoHists.meRECHIT_E_all = m_dbe->book1D("HO RecHit Energies","HO RecHit Energies",100,0,1000);
    hoHists.meRECHIT_T_tot = m_dbe->book1D("HO RecHit Times","HO RecHit Times",100,0,500);
    hoHists.meOCC_MAP_GEO = m_dbe->book2D("HO RecHit Geo Occupancy Map","RecHit Geo Occupancy Map",59,-29.5,29.5,74,0,73);

  }

  return;
}

void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  if(!m_dbe) { printf("HcalRecHitMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  ievt_++;
  meEVT_->Fill(ievt_);

  HBHERecHitCollection::const_iterator _ib;
  HORecHitCollection::const_iterator _io;
  HFRecHitCollection::const_iterator _if;
  float tot = 0; float all =0;
  if(hbHits.size()>0){
    for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits
      hbHists.meRECHIT_E_all->Fill(_ib->energy());
      if(_ib->energy()>occThresh_){
	hbHists.meOCC_MAP_GEO->Fill(_ib->id().ieta(),_ib->id().ieta());
	meOCC_MAP_all_GEO->Fill(_ib->id().ieta(),_ib->id().ieta());      
      }
      tot += _ib->energy();
      hbHists.meRECHIT_T_tot->Fill(_ib->time());
      if(doPerChannel_) HcalRecHitPerChan::perChanHists<HBHERecHit>(0,*_ib,hbHists.meRECHIT_E,hbHists.meRECHIT_T,m_dbe);
    }
    hbHists.meRECHIT_E_tot->Fill(tot);
    all += tot;
  }

  tot = 0;
  if(hoHits.size()>0){
    for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) { // loop over all hits
      hoHists.meRECHIT_E_all->Fill(_io->energy());
      if(_io->energy()>occThresh_){
	hoHists.meOCC_MAP_GEO->Fill(_io->id().ieta(),_io->id().ieta());
	meOCC_MAP_all_GEO->Fill(_io->id().ieta(),_io->id().ieta());      
      }
      tot += _io->energy();
      hoHists.meRECHIT_T_tot->Fill(_io->time());
      if(doPerChannel_) HcalRecHitPerChan::perChanHists<HORecHit>(1,*_io,hoHists.meRECHIT_E,hoHists.meRECHIT_T,m_dbe);
    }
    hoHists.meRECHIT_E_tot->Fill(tot);
    all += tot;
  }

  tot=0;
  if(hfHits.size()>0){
    for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) { // loop over all hits
      hfHists.meRECHIT_E_all->Fill(_if->energy());
      if(_if->energy()>occThresh_){
	hfHists.meOCC_MAP_GEO->Fill(_if->id().ieta(),_if->id().ieta());
	meOCC_MAP_all_GEO->Fill(_if->id().ieta(),_if->id().ieta());      
      }
      tot += _if->energy();
      hfHists.meRECHIT_T_tot->Fill(_if->time());
      if(doPerChannel_) HcalRecHitPerChan::perChanHists<HFRecHit>(2,*_if,hfHists.meRECHIT_E,hfHists.meRECHIT_T,m_dbe);
    }
    hfHists.meRECHIT_E_tot->Fill(tot);
    all += tot;
  }

  if(all>0) meRECHIT_E_all->Fill(all);
  return;
}

