#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"

HcalDigiMonitor::HcalDigiMonitor() {}

HcalDigiMonitor::~HcalDigiMonitor() {}

static bool bitUpset(int last, int now){
  if(last ==-1) return false;
  int v = last+1; if(v==4) v=0;
  if(v==now) return false;
  return true;
}

namespace HcalDigiErrs{
  template<class Digi>
  inline void fillErrors(const Digi& digi, MonitorElement* mapG, MonitorElement* mapE){
    if(digiErr(digi)){
      mapG->Fill(digi.id().ieta(),digi.id().iphi());
      mapE->Fill(digi.elecId().readoutVMECrateId(),digi.elecId().htrSlot());
    }
    return;
  }
  template<class Digi>
  static bool digiErr(const Digi& digi){
    int last = -1;
    for (int i=0; i<digi.size(); i++) { 
      if(bitUpset(last,digi.sample(i).capid())) return true;
      if(digi.sample(i).er()) return true;
    }
    return false;
  }
}

void HcalDigiMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/DigiMonitor/HBHE");
    hbHists.DIGI_NUM =  m_dbe->book1D("HB/HE # of Digis","HB/HE # of Digis",100,0,1000);
    hbHists.DIGI_SIZE =  m_dbe->book1D("HB/HE Digi Size","HB/HE Digi Size",100,0,100);
    hbHists.DIGI_PRESAMPLE =  m_dbe->book1D("HB/HE Digi Presamples","HB/HE Digi Presamples",100,0,100);
    hbHists.QIE_CAPID =  m_dbe->book1D("HB/HE QIE Cap-ID","HB/HE QIE Cap-ID",6,-0.5,5.5);
    hbHists.QIE_ADC = m_dbe->book1D("HB/HE QIE ADC Value","HB/HE QIE ADC Value",100,0,1000);
    hbHists.QIE_DV = m_dbe->book1D("HB/HE QIE Data Value","HB/HE QIE Data Value",2,-0.5,1.5);
    hbHists.ERR_MAP_GEO = m_dbe->book2D("HB/HE Digi Geo Error Map","HB/HE Digi Geo Error Map",59,-29.5,29.5,40,0,40);
    hbHists.ERR_MAP_ELEC = m_dbe->book2D("HB/HE Digi Elec Error Map","HB/HE Digi Elec Error Map",10,0,10,10,0,10);

    m_dbe->setCurrentFolder("Hcal/DigiMonitor/HF");
    hfHists.DIGI_NUM =  m_dbe->book1D("HF # of Digis","HF # of Digis",100,0,1000);
    hfHists.DIGI_SIZE =  m_dbe->book1D("HF Digi Size","HF Digi Size",100,0,100);
    hfHists.DIGI_PRESAMPLE =  m_dbe->book1D("HF Digi Presamples","HF Digi Presamples",100,0,100);
    hfHists.QIE_CAPID =  m_dbe->book1D("HF QIE Cap-ID","HF QIE Cap-ID",6,-0.5,5.5);
    hfHists.QIE_ADC = m_dbe->book1D("HF QIE ADC Value","HF QIE ADC Value",100,0,1000);
    hfHists.QIE_DV = m_dbe->book1D("HF QIE Data Value","HF QIE Data Value",2,-0.5,1.5);
    hfHists.ERR_MAP_GEO = m_dbe->book2D("HF Digi Geo Error Map","HF Digi Geo Error Map",59,-29.5,29.5,40,0,40);
    hfHists.ERR_MAP_ELEC = m_dbe->book2D("HF Digi Elec Error Map","HF Digi Elec Error Map",10,0,10,10,0,10);

    m_dbe->setCurrentFolder("Hcal/DigiMonitor/HO");
    hoHists.DIGI_NUM =  m_dbe->book1D("HO # of Digis","HO # of Digis",100,0,1000);
    hoHists.DIGI_SIZE =  m_dbe->book1D("HO Digi Size","HO Digi Size",100,0,100);
    hoHists.DIGI_PRESAMPLE =  m_dbe->book1D("HO Digi Presamples","HO Digi Presamples",100,0,100);
    hoHists.QIE_CAPID =  m_dbe->book1D("HO QIE Cap-ID","HO QIE Cap-ID",6,-0.5,5.5);
    hoHists.QIE_ADC = m_dbe->book1D("HO QIE ADC Value","HO QIE ADC Value",100,0,1000);
    hoHists.QIE_DV = m_dbe->book1D("HO QIE Data Value","HO QIE Data Value",2,-0.5,1.5);
     hoHists.ERR_MAP_GEO = m_dbe->book2D("HO Digi Geo Error Map","HO Digi Geo Error Map",59,-29.5,29.5,40,0,40);
    hoHists.ERR_MAP_ELEC = m_dbe->book2D("HO Digi Elec Error Map","HO Digi Elec Error Map",10,0,10,10,0,10);
  }

  return;
}

void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const HFDigiCollection& hf)
{

  if(!m_dbe) { printf("HcalDigiMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  try{
    hbHists.DIGI_NUM->Fill(hbhe.size());
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	
      HcalDigiErrs::fillErrors<HBHEDataFrame>(digi,hbHists.ERR_MAP_GEO,hbHists.ERR_MAP_ELEC);	  
      hbHists.DIGI_SIZE->Fill(digi.size());
      hbHists.DIGI_PRESAMPLE->Fill(digi.presamples());
      int last = -1;
      //    printf("hb/he digi crate: %d, %d\n",digi.elecId().readoutVMECrateId(),digi.elecId().htrSlot());
      for (int i=0; i<digi.size(); i++) {	    
	hbHists.QIE_CAPID->Fill(digi.sample(i).capid());
	hbHists.QIE_ADC->Fill(digi.sample(i).adc());
	hbHists.QIE_CAPID->Fill(5,bitUpset(last,digi.sample(i).capid()));
	last = digi.sample(i).capid();
	hbHists.QIE_DV->Fill(0,digi.sample(i).dv());
	hbHists.QIE_DV->Fill(1,digi.sample(i).er());
      }    
    }
  } catch (...) {
    printf("HcalDigiMonitor::processEvent  No HB/HE Digis.\n");
  }
  
  try{
     hoHists.DIGI_NUM->Fill(ho.size());
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);	
      HcalDigiErrs::fillErrors<HODataFrame>(digi,hoHists.ERR_MAP_GEO,hoHists.ERR_MAP_ELEC);  
      hoHists.DIGI_SIZE->Fill(digi.size());
      hoHists.DIGI_PRESAMPLE->Fill(digi.presamples());
      //     printf("ho digi crate: %d, %d\n",digi.elecId().readoutVMECrateId(),digi.elecId().htrSlot());
     int last = -1;
      for (int i=0; i<digi.size(); i++) {	    
	hoHists.QIE_CAPID->Fill(digi.sample(i).capid());
	hoHists.QIE_ADC->Fill(digi.sample(i).adc());
	hoHists.QIE_CAPID->Fill(5,bitUpset(last,digi.sample(i).capid()));
	last = digi.sample(i).capid();
	hoHists.QIE_DV->Fill(0,digi.sample(i).dv());
	hoHists.QIE_DV->Fill(1,digi.sample(i).er());
      }    
    }    
  } catch (...) {
    cout << "HcalDigiMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    hfHists.DIGI_NUM->Fill(hf.size());
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);	
      HcalDigiErrs::fillErrors<HFDataFrame>(digi,hfHists.ERR_MAP_GEO,hfHists.ERR_MAP_ELEC); 
      hfHists.DIGI_SIZE->Fill(digi.size());
      hfHists.DIGI_PRESAMPLE->Fill(digi.presamples());
      //    printf("hf digi crate: %d, %d\n",digi.elecId().readoutVMECrateId(),digi.elecId().htrSlot());
      int last = -1;
      for (int i=0; i<digi.size(); i++) {	    
	hfHists.QIE_CAPID->Fill(digi.sample(i).capid());
	hfHists.QIE_ADC->Fill(digi.sample(i).adc());
	hfHists.QIE_CAPID->Fill(5,bitUpset(last,digi.sample(i).capid()));
	last = digi.sample(i).capid();
	hfHists.QIE_DV->Fill(0,digi.sample(i).dv());
	hfHists.QIE_DV->Fill(1,digi.sample(i).er());
      }
    }
  } catch (...) {
    cout << "HcalDigiMonitor::processEvent  No HF Digis." << endl;
  }

}
