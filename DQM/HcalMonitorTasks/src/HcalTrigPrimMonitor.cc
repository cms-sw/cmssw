#include "DQM/HcalMonitorTasks/interface/HcalTrigPrimMonitor.h"

HcalTrigPrimMonitor::HcalTrigPrimMonitor() {
  ievt_=0;
}

HcalTrigPrimMonitor::~HcalTrigPrimMonitor() {
}

void HcalTrigPrimMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/TrigPrimMonitor");
    m_dbe->removeContents();
    meEVT_= 0;
  }

}

void HcalTrigPrimMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 41.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -41.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "TrigPrim eta min/max set to " << etaMin_ << "/" << etaMax_ << endl;
  
  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "TrigPrim phi min/max set to " << phiMin_ << "/" << phiMax_ << endl;
  
  ievt_=0;
  
  if ( m_dbe !=NULL ) {    

    m_dbe->setCurrentFolder("HcalMonitor/TrigPrimMonitor");
    meEVT_ = m_dbe->bookInt("TrigPrim Event Number");    
    meEVT_->Fill(ievt_);
  }

  return;
}

void HcalTrigPrimMonitor::processEvent(const HBHERecHitCollection& hbHits, 
				       const HORecHitCollection& hoHits, 
				       const HFRecHitCollection& hfHits,
				       const HcalTrigPrimDigiCollection& tpDigis){


  if(!m_dbe) { printf("HcalTrigPrimMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  ievt_++;
  meEVT_->Fill(ievt_);

   try{
    for (HcalTrigPrimDigiCollection::const_iterator j=tpDigis.begin(); j!=tpDigis.end(); j++){
      const HcalTriggerPrimitiveDigi digi = (const HcalTriggerPrimitiveDigi)(*j);
      if(digi.SOI_compressedEt()>0){
        std::cout << "size  " <<  digi.size() << std::endl;
	std::cout << "iphi  " <<  digi.id().iphi() << std::endl;
	std::cout << "ieta  " <<  digi.id().ieta() << std::endl;
	std::cout << "subdet  " <<  digi.id().subdet() << std::endl;
	std::cout << "zside  " <<  digi.id().zside() << std::endl;
	std::cout << "compressed Et  " <<  digi.SOI_compressedEt() << std::endl;
	std::cout << "FG bit  " <<  digi.SOI_fineGrain() << std::endl;
	std::cout << "raw  " <<  digi.t0().raw() << std::endl;
	std::cout << "raw Et " <<  digi.t0().compressedEt() << std::endl;
	std::cout << "raw FG " <<  digi.t0().fineGrain() << std::endl;
	std::cout << "raw slb " <<  digi.t0().slb() << std::endl;
	std::cout << "raw slbChan " <<  digi.t0().slbChan() << std::endl;
	std::cout << "raw slbAndChan " <<  digi.t0().slbAndChan() << std::endl;
      }
    }
   } catch (...) {    
     printf("HcalTrigPrimMonitor:  no tp digis\n");
  }


  return;
}

