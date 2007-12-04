#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include "TH1F.h"


HcalPedestalMonitor::HcalPedestalMonitor() {m_doPerChannel = false;}

HcalPedestalMonitor::~HcalPedestalMonitor() {}

void HcalPedestalMonitor::clearME(){
  
  if ( m_dbe ) {
    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HBHE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HF");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HO");
    m_dbe->removeContents();
  }

}
void HcalPedestalMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);

  if ( ps.getUntrackedParameter<bool>("PedestalsPerChannel", false) ) {
    m_doPerChannel = true;
  }

  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);

  ievt_=0;

  if ( m_dbe ) {
    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor");
    meEVT_ = m_dbe->bookInt("Pedestal Task Event Number");
    meEVT_->Fill(ievt_);

    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HBHE");
    hbHists.ALLPEDS =  m_dbe->book1D("HBHE All Pedestal Values","HBHE All Pedestal Values",15,0,14);
    hbHists.PEDRMS  =  m_dbe->book1D("HBHE Pedestal RMS Values","HBHE Pedestal RMS Values",100,0,3);
    hbHists.PEDMEAN =  m_dbe->book1D("HBHE Pedestal Mean Values","HBHE Pedestal Mean Values",100,0,9);
    hbHists.CAPIDRMS  =  m_dbe->book1D("HBHE CapID RMS Variance","HBHE CapID RMS Variance",50,0,0.5);
    hbHists.CAPIDMEAN =  m_dbe->book1D("HBHE CapID Mean Variance","HBHE CapID Mean Variance",50,0,3);
    hbHists.QIERMS  =  m_dbe->book1D("HBHE QIE RMS Values","HBHE QIE RMS Values",50,0,3);
    hbHists.QIEMEAN =  m_dbe->book1D("HBHE QIE Mean Values","HBHE QIE Mean Values",50,0,3);
    hbHists.ERRGEO =  m_dbe->book2D("HBHE Pedestal Geo Error Map","HBHE Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.ERRELEC =  m_dbe->book2D("HBHE Pedestal Elec Error Map","HBHE Pedestal Elec Error Map",20,0,20,20,0,20);

    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HF");
    hfHists.ALLPEDS =  m_dbe->book1D("HF All Pedestal Values","HF All Pedestal Values",15,0,14);
    hfHists.PEDRMS  =  m_dbe->book1D("HF Pedestal RMS Values","HF Pedestal RMS Values",100,0,3);
    hfHists.PEDMEAN =  m_dbe->book1D("HF Pedestal Mean Values","HF Pedestal Mean Values",100,0,9);
    hfHists.CAPIDRMS  =  m_dbe->book1D("HF CapID RMS Variance","HF CapID RMS Variance",50,0,0.5);
    hfHists.CAPIDMEAN =  m_dbe->book1D("HF CapID Mean Variance","HF CapID Mean Variance",50,0,3);
    hfHists.QIERMS  =  m_dbe->book1D("HF QIE RMS Values","HF QIE RMS Values",50,0,3);
    hfHists.QIEMEAN =  m_dbe->book1D("HF QIE Mean Values","HF QIE Mean Values",50,0,3);
    hfHists.ERRGEO =  m_dbe->book2D("HF Pedestal Geo Error Map","HF Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.ERRELEC =  m_dbe->book2D("HF Pedestal Elec Error Map","HF Pedestal Elec Error Map",20,0,20,20,0,20);

    m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HO");
    hoHists.ALLPEDS =  m_dbe->book1D("HO All Pedestal Values","HO All Pedestal Values",15,0,14);
    hoHists.PEDRMS  =  m_dbe->book1D("HO Pedestal RMS Values","HO Pedestal RMS Values",100,0,3);
    hoHists.PEDMEAN =  m_dbe->book1D("HO Pedestal Mean Values","HO Pedestal Mean Values",100,0,9);
    hoHists.CAPIDRMS  =  m_dbe->book1D("HO CapID RMS Variance","HO CapID RMS Variance",50,0,0.5);
    hoHists.CAPIDMEAN =  m_dbe->book1D("HO CapID Mean Variance","HO CapID Mean Variance",50,0,3);
    hoHists.QIERMS  =  m_dbe->book1D("HO QIE RMS Values","HO QIE RMS Values",50,0,3);
    hoHists.QIEMEAN =  m_dbe->book1D("HO QIE Mean Values","HO QIE Mean Values",50,0,3);
    hoHists.ERRGEO =  m_dbe->book2D("HO Pedestal Geo Error Map","HO Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.ERRELEC =  m_dbe->book2D("HO Pedestal Elec Error Map","HO Pedestal Elec Error Map",20,0,20,20,0,20);
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
				       const HcalDbService& conditions){
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  if(m_doPerChannel) m_shape = conditions.getHcalShape();
  
  if(!m_dbe) { printf("HcalPedestalMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }

  try{
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      m_coder = conditions.getHcalCoder(digi.id());
      for (int i=0; i<digi.size(); i++) {
	hbHists.ALLPEDS->Fill(digi.sample(i).adc());
	if(m_doPerChannel && digi.sample(i).adc()>0) perChanHists(0,digi.id(),digi.sample(i),hbHists.PEDVALS);
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
	if(m_doPerChannel && digi.sample(i).adc()>0) perChanHists(1,digi.id(),digi.sample(i),hoHists.PEDVALS);
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
	if(m_doPerChannel && digi.sample(i).adc()>0) perChanHists(2,digi.id(),digi.sample(i),hfHists.PEDVALS);
      }
    }
  } catch (...) {
    cout << "HcalPedestalMonitor::processEvent  No HF Digis." << endl;
  }

  return;
}

void HcalPedestalMonitor::done(){

  return;
}

void HcalPedestalMonitor::perChanHists(int id, const HcalDetId detid, const HcalQIESample& qie, map<HcalDetId, map<int, MonitorElement*> > &tool) {
  static const int bins=10;
  //  map<int,MonitorElement*> _mei;
  
  string type = "HBHE";
  if(m_dbe) m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HBHE");
  if(id==1) { 
    type = "HO"; 
    if(m_dbe) m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HO");
  }
  else if(id==2) { 
    type = "HF"; 
    if(m_dbe) m_dbe->setCurrentFolder("HcalMonitor/PedestalMonitor/HF");
  }  
  
  //outer iteration
  // map<int, MonitorElement*> it = tool[detid];
  bool gotit=false;
  if(REG[detid]) gotit=true;
  //  _meo = tool.find(detid);
  //  if (_meo!=tool.end()){
  if(gotit){
    //inner iteration
    map<int, MonitorElement*> _mei = tool[detid];
    //    _mei = _meo->second;
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
    REG[detid] = true;
  }
}
