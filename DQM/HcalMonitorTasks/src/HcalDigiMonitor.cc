#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"

HcalDigiMonitor::HcalDigiMonitor() {}

HcalDigiMonitor::~HcalDigiMonitor() {}

void HcalDigiMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/DigiMonitor/HBHE");
    m_meDIGI_SIZE_hb =  m_dbe->book1D("HB/HE Digi Size","HB/HE Digi Size",100,0,100);
    m_meDIGI_PRESAMPLE_hb =  m_dbe->book1D("HB/HE Digi Presamples","HB/HE Digi Presamples",100,0,100);
    m_meQIE_CAPID_hb =  m_dbe->book1D("HB/HE QIE Cap-ID","HB/HE QIE Cap-ID",6,-0.5,5.5);
    m_meQIE_ADC_hb = m_dbe->book1D("HB/HE QIE ADC Value","HB/HE QIE ADC Value",100,0,100);
    m_meQIE_DV_hb = m_dbe->book1D("HB/HE QIE Data Value","HB/HE QIE Data Value",2,-0.5,1.5);
    m_meERR_MAP_hb = m_dbe->book2D("HB/HE Digi Errors","HB/HE Digi Errors",59,-29.5,29.5,40,0,40);

    m_dbe->setCurrentFolder("Hcal/DigiMonitor/HF");
    m_meDIGI_SIZE_hf =  m_dbe->book1D("HF Digi Size","HF Digi Size",100,0,100);
    m_meDIGI_PRESAMPLE_hf =  m_dbe->book1D("HF Digi Presamples","HF Digi Presamples",100,0,100);
    m_meQIE_CAPID_hf =  m_dbe->book1D("HF QIE Cap-ID","HF QIE Cap-ID",6,-0.5,5.5);
    m_meQIE_ADC_hf = m_dbe->book1D("HF QIE ADC Value","HF QIE ADC Value",100,0,100);
    m_meQIE_DV_hf = m_dbe->book1D("HF QIE Data Value","HF QIE Data Value",2,-0.5,1.5);
    m_meERR_MAP_hf = m_dbe->book2D("HF Digi Errors","HF Digi Errors",59,-29.5,29.5,40,0,40);

    m_dbe->setCurrentFolder("Hcal/DigiMonitor/HO");
    m_meDIGI_SIZE_ho =  m_dbe->book1D("HO Digi Size","HO Digi Size",100,0,100);
    m_meDIGI_PRESAMPLE_ho =  m_dbe->book1D("HO Digi Presamples","HO Digi Presamples",100,0,100);
    m_meQIE_CAPID_ho =  m_dbe->book1D("HO QIE Cap-ID","HO QIE Cap-ID",6,-0.5,5.5);
    m_meQIE_ADC_ho = m_dbe->book1D("HO QIE ADC Value","HO QIE ADC Value",100,0,100);
    m_meQIE_DV_ho = m_dbe->book1D("HO QIE Data Value","HO QIE Data Value",2,-0.5,1.5);
    m_meERR_MAP_ho = m_dbe->book2D("HO Digi Errors","HO Digi Errors",59,-29.5,29.5,40,0,40);

  }

  return;
}

void HcalDigiMonitor::done(int mode){

}

bool bitUpset(int last, int now){
  if(last ==-1) return false;
  int v = last+1; if(v==4) v=0;
  if(v==now) return false;
  return true;
}

bool hbheErr(HBHEDataFrame digi){
  int last = -1;
  for (int i=0; i<digi.size(); i++) { 
    if(bitUpset(last,digi.sample(i).capid())) return true;
    if(digi.sample(i).er()) return true;
  }
  return false;
}

void HcalDigiMonitor::fillErrors(HBHEDataFrame digi){
  if(hbheErr(digi)){
    int x = digi.id().ieta();
    int y = digi.id().iphi();
    m_meERR_MAP_hb->Fill(x,y);
  }
  return;
}

bool hoErr(HODataFrame digi){
  int last = -1;
  for (int i=0; i<digi.size(); i++) { 
    if(bitUpset(last,digi.sample(i).capid())) return true;
    if(digi.sample(i).er()) return true;
  }
  return false;
}

void HcalDigiMonitor::fillErrors(HODataFrame digi){
  if(hoErr(digi)){
    int x = digi.id().ieta();
    int y = digi.id().iphi();
    m_meERR_MAP_ho->Fill(x,y);
  }
  return;
}

bool hfErr(HFDataFrame digi){
  int last = -1;
  for (int i=0; i<digi.size(); i++) { 
    if(bitUpset(last,digi.sample(i).capid())) return true;
    if(digi.sample(i).er()) return true;
  }
  return false;
}

void HcalDigiMonitor::fillErrors(HFDataFrame digi){
  if(hfErr(digi)){
    int x = digi.id().ieta();
    int y = digi.id().iphi();
    m_meERR_MAP_hf->Fill(x,y);
  }
  return;
}

void HcalDigiMonitor::processEvent(std::vector<edm::Handle<HBHEDigiCollection> > hbhe,
				   std::vector<edm::Handle<HODigiCollection> > ho,
				   std::vector<edm::Handle<HFDigiCollection> > hf)
{

  if(!m_dbe) { printf("HcalDigiMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }


  try{
    std::vector<edm::Handle<HBHEDigiCollection> >::iterator i;
    for (i=hbhe.begin(); i!=hbhe.end(); i++) {
      const HBHEDigiCollection& c=*(*i);
      for (HBHEDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++){
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);	
	fillErrors(digi);	  
	m_meDIGI_SIZE_hb->Fill(digi.size());
	m_meDIGI_PRESAMPLE_hb->Fill(digi.presamples());
	int last = -1;
	for (int i=0; i<digi.size(); i++) {	    
	  m_meQIE_CAPID_hb->Fill(digi.sample(i).capid());
	  m_meQIE_ADC_hb->Fill(digi.sample(i).adc());
	  m_meQIE_CAPID_hb->Fill(5,bitUpset(last,digi.sample(i).capid()));
	  if(bitUpset(last,digi.sample(i).capid())) printf("l: %d, n: %d\n",last,digi.sample(i).capid());
	  last = digi.sample(i).capid();
	  m_meQIE_DV_hb->Fill(0,digi.sample(i).dv());
	  m_meQIE_DV_hb->Fill(1,digi.sample(i).er());
	}
      }
    }
  } catch (...) {
    printf("HcalDigiMonitor::processEvent  No HB/HE Digis.\n");
  }

  try{
    std::vector<edm::Handle<HODigiCollection> >::iterator i;

    for (i=ho.begin(); i!=ho.end(); i++) {
      const HODigiCollection& c=*(*i);
      for (HODigiCollection::const_iterator j=c.begin(); j!=c.end(); j++){
	const HODataFrame digi = (const HODataFrame)(*j);	
	fillErrors(digi);	  
	m_meDIGI_SIZE_ho->Fill(digi.size());
	m_meDIGI_PRESAMPLE_ho->Fill(digi.presamples());
	int last = -1;
	for (int i=0; i<digi.size(); i++) {	    
	  m_meQIE_CAPID_ho->Fill(digi.sample(i).capid());
	  m_meQIE_ADC_ho->Fill(digi.sample(i).adc());
	  m_meQIE_CAPID_ho->Fill(5,bitUpset(last,digi.sample(i).capid()));
	  if(bitUpset(last,digi.sample(i).capid())) printf("l: %d, n: %d\n",last,digi.sample(i).capid());
	  last = digi.sample(i).capid();
	  m_meQIE_DV_ho->Fill(0,digi.sample(i).dv());
	  m_meQIE_DV_ho->Fill(1,digi.sample(i).er());
	}
      }
    }
  } catch (...) {
    cout << "HcalDigiMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    std::vector<edm::Handle<HFDigiCollection> >::iterator i;
    for (i=hf.begin(); i!=hf.end(); i++) {
      const HFDigiCollection& c=*(*i);
      for (HFDigiCollection::const_iterator j=c.begin(); j!=c.end(); j++){
	const HFDataFrame digi = (const HFDataFrame)(*j);	
	fillErrors(digi);	  
	m_meDIGI_SIZE_hf->Fill(digi.size());
	m_meDIGI_PRESAMPLE_hf->Fill(digi.presamples());
	int last = -1;
	for (int i=0; i<digi.size(); i++) {	    
	  m_meQIE_CAPID_hf->Fill(digi.sample(i).capid());
	  m_meQIE_ADC_hf->Fill(digi.sample(i).adc());
	  m_meQIE_CAPID_hf->Fill(5,bitUpset(last,digi.sample(i).capid()));
	  if(bitUpset(last,digi.sample(i).capid())) printf("l: %d, n: %d\n",last,digi.sample(i).capid());
	  last = digi.sample(i).capid();
	  m_meQIE_DV_hf->Fill(0,digi.sample(i).dv());
	  m_meQIE_DV_hf->Fill(1,digi.sample(i).er());
	}
      }
    }
  } catch (...) {
    cout << "HcalDigiMonitor::processEvent  No HF Digis." << endl;
  }

}
