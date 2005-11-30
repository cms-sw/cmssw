#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"

HcalRecHitMonitor::HcalRecHitMonitor() {
  m_doPerChannel = false;
}

HcalRecHitMonitor::~HcalRecHitMonitor() {}


namespace HcalRecHitPerChan{
  template<class RecHit>
  inline void perChanHists(int id, const RecHit& rhit, std::map<HcalDetId, MonitorElement*> &toolE, std::map<HcalDetId, MonitorElement*> &toolT, DaqMonitorBEInterface* dbe) {
    
    std::map<HcalDetId,MonitorElement*>::iterator _mei;
    string type = "HB/HE";
    if(dbe) dbe->setCurrentFolder("Hcal/RecHitMonitor/HBHE");
    if(id==0) { 
      type = "HB/HE"; 
      if(dbe) dbe->setCurrentFolder("Hcal/RecHitMonitor/HBHE");
    }
    else if(id==1) { 
      type = "HO"; 
      if(dbe) dbe->setCurrentFolder("Hcal/RecHitMonitor/HO");
    }
    else if(id==2) { 
      type = "HF"; 
      if(dbe) dbe->setCurrentFolder("Hcal/RecHitMonitor/HF");
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
	sprintf(name,"%s RecHit Energy, ieta=%d iphi=%d depth=%d",type.c_str(),rhit.id().ieta(),rhit.id().iphi(),rhit.id().depth());
	toolE[rhit.id()] =  dbe->book1D(name,name,100,0,1000); 
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
	sprintf(name,"%s RecHit Time, ieta=%d iphi=%d depth=%d",type.c_str(),rhit.id().ieta(),rhit.id().iphi(),rhit.id().depth());
	toolT[rhit.id()] =  dbe->book1D(name,name,100,0,1000); 
	toolT[rhit.id()]->Fill(rhit.time());
      }
    }
  }
}


void HcalRecHitMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ) {
    m_doPerChannel = true;
  }

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/RecHitMonitor/HBHE");
    m_meRECHIT_E_hb_tot =  m_dbe->book1D("HB/HE RecHit Energies","HB/HE RecHit Energies",100,0,10000);
    m_meRECHIT_T_hb_tot =  m_dbe->book1D("HB/HE RecHit Times","HB/HE RecHit Times",100,0,1000);

    m_dbe->setCurrentFolder("Hcal/RecHitMonitor/HF");
    m_meRECHIT_E_hf_tot =  m_dbe->book1D("HF RecHit Energies","HF RecHit Energies",100,0,10000);
    m_meRECHIT_T_hf_tot =  m_dbe->book1D("HF RecHit Times","HF RecHit Times",100,0,1000);

    m_dbe->setCurrentFolder("Hcal/RecHitMonitor/HO");
    m_meRECHIT_E_ho_tot =  m_dbe->book1D("HO RecHit Energies","HO RecHit Energies",100,0,10000);
    m_meRECHIT_T_ho_tot =  m_dbe->book1D("HO RecHit Times","HO RecHit Times",100,0,1000);

  }

  return;
}

void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  HBHERecHitCollection::const_iterator _ib;
  HORecHitCollection::const_iterator _io;
  HFRecHitCollection::const_iterator _if;

  for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits
    //    if(fabs(_ib->id().ieta())>14) continue;
    m_meRECHIT_E_hb_tot->Fill(_ib->energy());
    m_meRECHIT_T_hb_tot->Fill(_ib->time());
    if(m_doPerChannel) HcalRecHitPerChan::perChanHists<HBHERecHit>(0,*_ib,m_meRECHIT_E_hb,m_meRECHIT_T_hb,m_dbe);
  }
  

  for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) { // loop over all hits
    //    if(fabs(_io->id().ieta())>14) continue;
    m_meRECHIT_E_ho_tot->Fill(_io->energy());
    m_meRECHIT_T_ho_tot->Fill(_io->time());
    if(m_doPerChannel) HcalRecHitPerChan::perChanHists<HORecHit>(1,*_io,m_meRECHIT_E_ho,m_meRECHIT_T_ho,m_dbe);
  }
  
  for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) { // loop over all hits
    //    if(fabs(_if->id().ieta())>14) continue;
    m_meRECHIT_E_hf_tot->Fill(_if->energy());
    m_meRECHIT_T_hf_tot->Fill(_if->time());
    if(m_doPerChannel) HcalRecHitPerChan::perChanHists<HFRecHit>(2,*_if,m_meRECHIT_E_hf,m_meRECHIT_T_hf,m_dbe);
  }
  
  return;
}

