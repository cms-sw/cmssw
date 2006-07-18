#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include "TH1F.h"


HcalPedestalMonitor::HcalPedestalMonitor() {m_doPerChannel = false;}

HcalPedestalMonitor::~HcalPedestalMonitor() {}

void HcalPedestalMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( ps.getUntrackedParameter<bool>("PedestalsPerChannel", false) ) {
    m_doPerChannel = true;
  }

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal/PedestalMonitor/HBHE");
    hbHists.ALLPEDS =  m_dbe->book1D("HBHE All Pedestal Values","HBHE All Pedestal Values",15,0,14);
    m_dbe->setCurrentFolder("Hcal/PedestalMonitor/HF");
    hfHists.ALLPEDS =  m_dbe->book1D("HF All Pedestal Values","HF All Pedestal Values",15,0,14);
    m_dbe->setCurrentFolder("Hcal/PedestalMonitor/HO");
    hoHists.ALLPEDS =  m_dbe->book1D("HO All Pedestal Values","HO All Pedestal Values",15,0,14);
  }

  m_outputFile = ps.getUntrackedParameter<string>("PedestalFile", "");
  if ( m_outputFile.size() != 0 ) {
    cout << "Hcal Pedestal Calibrations will be saved to " << m_outputFile.c_str() << endl;
  }

  return;
}

void HcalPedestalMonitor::processEvent(const HBHEDigiCollection& hbhe,
				       const HODigiCollection& ho,
				       const HFDigiCollection& hf,
				       const HcalDbService& conditions)
{
  
  m_shape = conditions.getHcalShape();

  if(!m_dbe) { printf("HcalPedestalMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  try{
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      m_coder = conditions.getHcalCoder(digi.id());
      for (int i=0; i<digi.size(); i++) {
	hbHists.ALLPEDS->Fill(digi.sample(i).adc());
	if(m_doPerChannel) perChanHists(0,digi.id(),digi.sample(i),hbHists.PEDVALS);
      }
    }
  } catch (...) {
    printf("HcalPedestalMonitor::processEvent  No HBHE Digis.\n");
  }
  
  try{
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);	
      m_coder = conditions.getHcalCoder(digi.id());
      for (int i=0; i<digi.size(); i++) {
	hoHists.ALLPEDS->Fill(digi.sample(i).adc());
	if(m_doPerChannel) perChanHists(1,digi.id(),digi.sample(i),hoHists.PEDVALS);
      }
    }        
  } catch (...) {
    cout << "HcalPedestalMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);	
      m_coder = conditions.getHcalCoder(digi.id());
      for (int i=0; i<digi.size(); i++) {
	hfHists.ALLPEDS->Fill(digi.sample(i).adc());
	if(m_doPerChannel) perChanHists(2,digi.id(),digi.sample(i),hfHists.PEDVALS);
      }
    }
  } catch (...) {
    cout << "HcalPedestalMonitor::processEvent  No HF Digis." << endl;
  }

}

void HcalPedestalMonitor::done(){

  return;
}

void HcalPedestalMonitor::perChanHists(int id, const HcalDetId detid, const HcalQIESample& qie, map<HcalDetId, map<int, MonitorElement*> > &tool) {
  static const int bins=10;
  map<int,MonitorElement*> _mei;
  
  string type = "HBHE";
  if(m_dbe) m_dbe->setCurrentFolder("Hcal/PedestalMonitor/HBHE");
  if(id==1) { 
    type = "HO"; 
    if(m_dbe) m_dbe->setCurrentFolder("Hcal/PedestalMonitor/HO");
  }
  else if(id==2) { 
    type = "HF"; 
    if(m_dbe) m_dbe->setCurrentFolder("Hcal/PedestalMonitor/HF");
  }  

  ///outer iteration
  _meo = tool.find(detid);
  if (_meo!=tool.end()){
    //inner iteration
    _mei = _meo->second;
    if(_mei[qie.capid()]==NULL) printf("HcalPedestalAnalysis::perChanHists  This histo is NULL!!??\n");
    else if (qie.adc()<bins) _mei[qie.capid()]->Fill(qie.adc());
  }
  else{
    if(m_dbe){
      map<int,MonitorElement*> insert;
      float hi = 9; float lo = 0;
      for(int i=0; i<4; i++){
	char name[1024];
	sprintf(name,"%s Pedestal Value ieta=%d iphi=%d depth=%d CAPID=%d",type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i);      
	getLinearizedADC(*m_shape,m_coder,bins,i,lo,hi);
	insert[i] =  m_dbe->book1D(name,name,bins,lo,hi);
      }
      insert[qie.capid()]->Fill(qie.adc());
      tool[detid] = insert;
    }
  }
}
