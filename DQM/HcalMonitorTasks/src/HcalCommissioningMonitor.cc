#include "DQM/HcalMonitorTasks/interface/HcalCommissioningMonitor.h"

HcalCommissioningMonitor::HcalCommissioningMonitor() {
  ievt_=0;
}

HcalCommissioningMonitor::~HcalCommissioningMonitor() {}


void HcalCommissioningMonitor::reset(){}

void HcalCommissioningMonitor::clearME(){
   if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/CommissioingMonitor");
    m_dbe->removeContents();
    meEVT_= 0;
  }
}
void HcalCommissioningMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "MTCC eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "MTCC phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;
  
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    
    m_dbe->setCurrentFolder("HcalMonitor/CommissioningMonitor");
    
    meEVT_ = m_dbe->bookInt("Commissioning Event Number");    
    meEVT_->Fill(ievt_); 
  }

  return;
}

void HcalCommissioningMonitor::processEvent(const HBHEDigiCollection& hbhe,
					    const HODigiCollection& ho,
					    const HFDigiCollection& hf,
					    const HBHERecHitCollection& hbHits, 
					    const HORecHitCollection& hoHits,
					    const HFRecHitCollection& hfHits,
					    const LTCDigiCollection& ltc,
					    const HcalDbService& cond){ 
  
  if(!m_dbe) { printf("HcalCommissioningMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  return;
}
