#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"

HcalRecHitMonitor::HcalRecHitMonitor() {}

HcalRecHitMonitor::~HcalRecHitMonitor() {}

void HcalRecHitMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/RecHitMonitor/HBHE");
    m_meRECHIT_E_hb =  m_dbe->book1D("HB/HE RecHit Energy","HB/HE RecHit Energy",100,0,10000);
    m_meRECHIT_T_hb =  m_dbe->book1D("HB/HE RecHit Time","HB/HE RecHit Time",100,0,100);

    m_dbe->setCurrentFolder("Hcal/RecHitMonitor/HF");
    m_meRECHIT_E_hf =  m_dbe->book1D("HF RecHit Energy","HF RecHit Energy",100,0,10000);
    m_meRECHIT_T_hf =  m_dbe->book1D("HF RecHit Time","HF RecHit Time",100,0,100);

    m_dbe->setCurrentFolder("Hcal/RecHitMonitor/HO");
    m_meRECHIT_E_ho =  m_dbe->book1D("HO RecHit Energy","HO RecHit Energy",100,0,10000);
    m_meRECHIT_T_ho =  m_dbe->book1D("HO RecHit Time","HO RecHit Time",100,0,100);

  }

  return;
}

void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits){

  HBHERecHitCollection::const_iterator _ib;
  HORecHitCollection::const_iterator _io;
  HFRecHitCollection::const_iterator _if;

  for (_ib=hbHits.begin(); _ib!=hbHits.end(); _ib++) { // loop over all hits
    //    if(fabs(_ib->id().ieta())>14) continue;
    m_meRECHIT_E_hb->Fill(_ib->energy());
    m_meRECHIT_T_hb->Fill(_ib->time());
  }
  for (_io=hoHits.begin(); _io!=hoHits.end(); _io++) { // loop over all hits
    //    if(fabs(_io->id().ieta())>14) continue;
    m_meRECHIT_E_ho->Fill(_io->energy());
    m_meRECHIT_T_ho->Fill(_io->time());
  }
  for (_if=hfHits.begin(); _if!=hfHits.end(); _if++) { // loop over all hits
    //    if(fabs(_if->id().ieta())>14) continue;
    m_meRECHIT_E_hf->Fill(_if->energy());
    m_meRECHIT_T_hf->Fill(_if->time());
  }
  
  return;
}
