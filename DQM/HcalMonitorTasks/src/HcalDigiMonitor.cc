#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"

HcalDigiMonitor::HcalDigiMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1;
  ievt_=0;
}

HcalDigiMonitor::~HcalDigiMonitor() {
}

namespace HcalDigiPerChan{
  template<class Digi>
  inline void perChanHists(int id, const Digi& digi, vector<float> ampl,std::map<HcalDetId, MonitorElement*> &tool, DaqMonitorBEInterface* dbe) {
    
    std::map<HcalDetId,MonitorElement*>::iterator _mei;
    string type = "HB";
    if(dbe) dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HB");
    if(id==1) { 
      type = "HE"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HE");
    }
    else if(id==2) { 
      type = "HO"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HO");
    }
    else if(id==3) { 
      type = "HF"; 
      if(dbe) dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HF");
    }
    
    ///shapes by channel
    _mei=tool.find(digi.id()); // look for a histogram with this hit's id
    if (_mei!=tool.end()){
      if (_mei->second==0) cout << "HcalDigiMonitor::perChanHists, Found the histo, but it's null??";
      else{
	for (int i=0; i<digi.size(); i++) tool[digi.id()]->Fill(i,ampl[i]);
      }
    }
    else{
      if(dbe){
	char name[1024];
	sprintf(name,"%s Digi Shape ieta=%d iphi=%d depth=%d",type.c_str(),digi.id().ieta(),digi.id().iphi(),digi.id().depth());
	tool[digi.id()] =  dbe->book1D(name,name,11,-0.5,10.5); 
	for (int i=0; i<digi.size(); i++) tool[digi.id()]->Fill(i,ampl[i]);
      }
    }
  }
}

void HcalDigiMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HB");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HE");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HF");
    m_dbe->removeContents();
    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HO");
    m_dbe->removeContents();
  }
  return;
}

static bool bitUpset(int last, int now){
  if(last ==-1) return false;
  int v = last+1; if(v==4) v=0;
  if(v==now) return false;
  return true;
}

namespace HcalDigiMap{
  template<class Digi>
  inline void fillErrors(const Digi& digi, MonitorElement* mapGEO, MonitorElement* mapVME, 
			 MonitorElement* mapFIB, MonitorElement* mapDCC){
    if(digiErr(digi)){
      mapGEO->Fill(digi.id().ieta(),digi.id().iphi());
      mapVME->Fill(digi.elecId().readoutVMECrateId(),digi.elecId().htrSlot());
      mapFIB->Fill(digi.elecId().fiberChanId(),digi.elecId().fiberIndex());
      mapDCC->Fill(digi.elecId().spigot(),digi.elecId().dccid());
    }
    return;
  }

  template<class Digi>
  inline void fillOccupancy(const Digi& digi, MonitorElement* mapG1, MonitorElement* mapG2,
			    MonitorElement* mapG3, MonitorElement* mapG4,  
			    MonitorElement* mapVME, MonitorElement* mapFIB, MonitorElement* mapDCC, 
			    MonitorElement* mapEta, MonitorElement* mapPhi,
			    float thr){
    if(digiOccupied(digi,thr)){
      if(digi.id().depth()==1) mapG1->Fill(digi.id().ieta(),digi.id().iphi());
      if(digi.id().depth()==2) mapG2->Fill(digi.id().ieta(),digi.id().iphi());
      if(digi.id().depth()==3) mapG3->Fill(digi.id().ieta(),digi.id().iphi());
      if(digi.id().depth()==4) mapG4->Fill(digi.id().ieta(),digi.id().iphi());
      mapVME->Fill(digi.elecId().readoutVMECrateId(),digi.elecId().htrSlot());
      mapFIB->Fill(digi.elecId().fiberChanId(),digi.elecId().fiberIndex());
      mapDCC->Fill(digi.elecId().spigot(),digi.elecId().dccid());
      mapEta->Fill(digi.id().ieta());
      mapPhi->Fill(digi.id().iphi());
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

  template<class Digi>
  static bool digiOccupied(const Digi& digi, float thr){
    for (int i=0; i<digi.size(); i++) { 
      if(digi.sample(i).adc()>thr) return true;
    }
    return false;
  }
  
}

void HcalDigiMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  occThresh_ = ps.getUntrackedParameter<int>("DigiOccThresh", 10);
  cout << "Digi occupancy threshold set to " << occThresh_ << endl;

  if ( ps.getUntrackedParameter<bool>("DigisPerChannel", false) ) doPerChannel_ = true;  

  fedUnpackList_ = ps.getParameter<vector<int> >("FEDs");
  firstFED_ = ps.getParameter<int>("HcalFirstFED");

  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "Digi eta min/max set to " << etaMin_ << "/" <<etaMax_ << endl;

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", 0);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "Digi phi min/max set to " << phiMin_ << "/" <<phiMax_ << endl;

  ievt_=0;
  
  if ( m_dbe ) {

    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor");        
    meEVT_ = m_dbe->bookInt("Digi Task Event Number");    
    meEVT_->Fill(ievt_);
    
    OCC_L1 = m_dbe->book2D("Digi Depth 1 Occupancy Map","Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L2 = m_dbe->book2D("Digi Depth 2 Occupancy Map","Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L3 = m_dbe->book2D("Digi Depth 3 Occupancy Map","Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L4 = m_dbe->book2D("Digi Depth 4 Occupancy Map","Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_ETA = m_dbe->book1D("Digi Eta Occupancy Map","Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    OCC_PHI = m_dbe->book1D("Digi Phi Occupancy Map","Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    OCC_ELEC_VME = m_dbe->book2D("Digi VME Occupancy Map","Digi VME Occupancy Map",21,-0.5,20.5,21,-0.5,20.5);
    OCC_ELEC_FIB = m_dbe->book2D("Digi Fiber Occupancy Map","Digi Fiber Occupancy Map",3,-0.5,2.5,9,-0.5,8.5);
    OCC_ELEC_DCC = m_dbe->book2D("Digi Spigot Occupancy Map","Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());
    ERR_MAP_GEO = m_dbe->book2D("Digi Geo Error Map","Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    ERR_MAP_VME = m_dbe->book2D("Digi VME Error Map","Digi VME Error Map",21,-0.5,20.5,21,-0.5,20.5);
    ERR_MAP_FIB = m_dbe->book2D("Digi Fiber Error Map","Digi Fiber Error Map",3,-0.5,2.5,9,-0.5,8.5);
    ERR_MAP_DCC = m_dbe->book2D("Digi Spigot Error Map","Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());


    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HB");
    hbHists.SHAPE_tot =  m_dbe->book1D("HB Digi Shape","HB Digi Shape",11,-0.5,10.5);
    hbHists.SHAPE_THR_tot =  m_dbe->book1D("HB Digi Shape - over thresh","HB Digi Shape - over thresh",11,-0.5,10.5);
    hbHists.DIGI_NUM =  m_dbe->book1D("HB # of Digis","HB # of Digis",200,0,1000);
    hbHists.DIGI_SIZE =  m_dbe->book1D("HB Digi Size","HB Digi Size",50,0,50);
    hbHists.DIGI_PRESAMPLE =  m_dbe->book1D("HB Digi Presamples","HB Digi Presamples",50,0,50);
    hbHists.QIE_CAPID =  m_dbe->book1D("HB QIE Cap-ID","HB QIE Cap-ID",6,-0.5,5.5);
    hbHists.QIE_ADC = m_dbe->book1D("HB QIE ADC Value","HB QIE ADC Value",100,0,200);
    hbHists.QIE_DV = m_dbe->book1D("HB QIE Data Value","HB QIE Data Value",2,-0.5,1.5);
    
    hbHists.ERR_MAP_GEO = m_dbe->book2D("HB Digi Geo Error Map","HB Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.ERR_MAP_VME = m_dbe->book2D("HB Digi VME Error Map","HB Digi VME Error Map",21,-0.5,20.5,21,-0.5,20.5);
    hbHists.ERR_MAP_FIB = m_dbe->book2D("HB Digi Fiber Error Map","HB Digi Fiber Error Map",3,-0.5,2.5,9,-0.5,8.5);
    hbHists.ERR_MAP_DCC = m_dbe->book2D("HB Digi Spigot Error Map","HB Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());

    hbHists.OCC_MAP_GEO1 = m_dbe->book2D("HB Digi Depth 1 Occupancy Map","HB Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO2 = m_dbe->book2D("HB Digi Depth 2 Occupancy Map","HB Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO3 = m_dbe->book2D("HB Digi Depth 3 Occupancy Map","HB Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO4 = m_dbe->book2D("HB Digi Depth 4 Occupancy Map","HB Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_ETA = m_dbe->book1D("HB Digi Eta Occupancy Map","HB Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    hbHists.OCC_PHI = m_dbe->book1D("HB Digi Phi Occupancy Map","HB Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    hbHists.OCC_MAP_VME = m_dbe->book2D("HB Digi VME Occupancy Map","HB Digi VME Occupancy Map",21,-0.5,20.5,21,-0.5,20.5);
    hbHists.OCC_MAP_FIB = m_dbe->book2D("HB Digi Fiber Occupancy Map","HB Digi Fiber Occupancy Map",3,-0.5,2.5,9,-0.5,8.5);
    hbHists.OCC_MAP_DCC = m_dbe->book2D("HB Digi Spigot Occupancy Map","HB Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());
    
    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HE");
    heHists.SHAPE_tot =  m_dbe->book1D("HE Digi Shape","HE Digi Shape",11,-0.5,10.5);
    heHists.SHAPE_THR_tot =  m_dbe->book1D("HE Digi Shape - over thresh","HE Digi Shape - over thresh",11,-0.5,10.5);
    heHists.DIGI_NUM =  m_dbe->book1D("HE # of Digis","HE # of Digis",200,0,1000);
    heHists.DIGI_SIZE =  m_dbe->book1D("HE Digi Size","HE Digi Size",50,0,50);
    heHists.DIGI_PRESAMPLE =  m_dbe->book1D("HE Digi Presamples","HE Digi Presamples",50,0,50);
    heHists.QIE_CAPID =  m_dbe->book1D("HE QIE Cap-ID","HE QIE Cap-ID",6,-0.5,5.5);
    heHists.QIE_ADC = m_dbe->book1D("HE QIE ADC Value","HE QIE ADC Value",100,0,200);
    heHists.QIE_DV = m_dbe->book1D("HE QIE Data Value","HE QIE Data Value",2,-0.5,1.5);

    heHists.ERR_MAP_GEO = m_dbe->book2D("HE Digi Geo Error Map","HE Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.ERR_MAP_VME = m_dbe->book2D("HE Digi VME Error Map","HE Digi VME Error Map",21,-0.5,20.5,21,-0.5,20.5);
    heHists.ERR_MAP_FIB = m_dbe->book2D("HE Digi Fiber Error Map","HE Digi Fiber Error Map",3,-0.5,2.5,9,-0.5,8.5);
    heHists.ERR_MAP_DCC = m_dbe->book2D("HE Digi Spigot Error Map","HE Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());

    heHists.OCC_MAP_GEO1 = m_dbe->book2D("HE Digi Depth 1 Occupancy Map","HE Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO2 = m_dbe->book2D("HE Digi Depth 2 Occupancy Map","HE Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO3 = m_dbe->book2D("HE Digi Depth 3 Occupancy Map","HE Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO4 = m_dbe->book2D("HE Digi Depth 4 Occupancy Map","HE Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_ETA = m_dbe->book1D("HE Digi Eta Occupancy Map","HE Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    heHists.OCC_PHI = m_dbe->book1D("HE Digi Phi Occupancy Map","HE Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    heHists.OCC_MAP_VME = m_dbe->book2D("HE Digi VME Occupancy Map","HE Digi VME Occupancy Map",21,-0.5,20.5,21,-0.5,20.5);
    heHists.OCC_MAP_FIB = m_dbe->book2D("HE Digi Fiber Occupancy Map","HE Digi Fiber Occupancy Map",3,-0.5,2.5,9,-0.5,8.5);
    heHists.OCC_MAP_DCC = m_dbe->book2D("HE Digi Spigot Occupancy Map","HE Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());

    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HF");
    hfHists.SHAPE_tot =  m_dbe->book1D("HF Digi Shape","HF Digi Shape",11,-0.5,10.5);
    hfHists.SHAPE_THR_tot =  m_dbe->book1D("HF Digi Shape - over thresh","HF Digi Shape - over thresh",11,-0.5,10.5);
    hfHists.DIGI_NUM =  m_dbe->book1D("HF # of Digis","HF # of Digis",200,0,1000);
    hfHists.DIGI_SIZE =  m_dbe->book1D("HF Digi Size","HF Digi Size",50,0,50);
    hfHists.DIGI_PRESAMPLE =  m_dbe->book1D("HF Digi Presamples","HF Digi Presamples",50,0,50);
    hfHists.QIE_CAPID =  m_dbe->book1D("HF QIE Cap-ID","HF QIE Cap-ID",6,-0.5,5.5);
    hfHists.QIE_ADC = m_dbe->book1D("HF QIE ADC Value","HF QIE ADC Value",100,0,200);
    hfHists.QIE_DV = m_dbe->book1D("HF QIE Data Value","HF QIE Data Value",2,-0.5,1.5);

    hfHists.ERR_MAP_GEO = m_dbe->book2D("HF Digi Geo Error Map","HF Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.ERR_MAP_VME = m_dbe->book2D("HF Digi VME Error Map","HF Digi VME Error Map",21,-0.5,20.5,21,-0.5,20.5);
    hfHists.ERR_MAP_FIB = m_dbe->book2D("HF Digi Fiber Error Map","HF Digi Fiber Error Map",3,-0.5,2.5,9,-0.5,8.5);
    hfHists.ERR_MAP_DCC = m_dbe->book2D("HF Digi Spigot Error Map","HF Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());

    hfHists.OCC_MAP_GEO1 = m_dbe->book2D("HF Digi Depth 1 Occupancy Map","HF Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO2 = m_dbe->book2D("HF Digi Depth 2 Occupancy Map","HF Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO3 = m_dbe->book2D("HF Digi Depth 3 Occupancy Map","HF Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO4 = m_dbe->book2D("HF Digi Depth 4 Occupancy Map","HF Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_ETA = m_dbe->book1D("HF Digi Eta Occupancy Map","HF Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    hfHists.OCC_PHI = m_dbe->book1D("HF Digi Phi Occupancy Map","HF Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    hfHists.OCC_MAP_VME = m_dbe->book2D("HF Digi VME Occupancy Map","HF Digi VME Occupancy Map",21,-0.5,20.5,21,-0.5,20.5);
    hfHists.OCC_MAP_FIB = m_dbe->book2D("HF Digi Fiber Occupancy Map","HF Digi Fiber Occupancy Map",3,-0.5,2.5,9,-0.5,8.5);
    hfHists.OCC_MAP_DCC = m_dbe->book2D("HF Digi Spigot Occupancy Map","HF Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());

    m_dbe->setCurrentFolder("HcalMonitor/DigiMonitor/HO");
    hoHists.SHAPE_tot =  m_dbe->book1D("HO Digi Shape","HO Digi Shape",11,-0.5,10.5);
    hoHists.SHAPE_THR_tot =  m_dbe->book1D("HO Digi Shape - over thresh","HO Digi Shape - over thresh",11,-0.5,10.5);
    hoHists.DIGI_NUM =  m_dbe->book1D("HO # of Digis","HO # of Digis",200,0,1000);
    hoHists.DIGI_SIZE =  m_dbe->book1D("HO Digi Size","HO Digi Size",50,0,50);
    hoHists.DIGI_PRESAMPLE =  m_dbe->book1D("HO Digi Presamples","HO Digi Presamples",50,0,50);
    hoHists.QIE_CAPID =  m_dbe->book1D("HO QIE Cap-ID","HO QIE Cap-ID",6,-0.5,5.5);
    hoHists.QIE_ADC = m_dbe->book1D("HO QIE ADC Value","HO QIE ADC Value",100,0,200);
    hoHists.QIE_DV = m_dbe->book1D("HO QIE Data Value","HO QIE Data Value",2,-0.5,1.5);

    hoHists.ERR_MAP_GEO = m_dbe->book2D("HO Digi Geo Error Map","HO Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.ERR_MAP_VME = m_dbe->book2D("HO Digi VME Error Map","HO Digi VME Error Map",21,-0.5,20.5,21,-0.5,20.5);
    hoHists.ERR_MAP_FIB = m_dbe->book2D("HO Digi Fiber Error Map","HO Digi Fiber Error Map",3,-0.5,2.5,9,-0.5,8.5);
    hoHists.ERR_MAP_DCC = m_dbe->book2D("HO Digi Spigot Error Map","HO Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());

    hoHists.OCC_MAP_GEO1 = m_dbe->book2D("HO Digi Depth 1 Occupancy Map","HO Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO2 = m_dbe->book2D("HO Digi Depth 2 Occupancy Map","HO Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO3 = m_dbe->book2D("HO Digi Depth 3 Occupancy Map","HO Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO4 = m_dbe->book2D("HO Digi Depth 4 Occupancy Map","HO Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_ETA = m_dbe->book1D("HO Digi Eta Occupancy Map","HO Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    hoHists.OCC_PHI = m_dbe->book1D("HO Digi Phi Occupancy Map","HO Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);

    hoHists.OCC_MAP_VME = m_dbe->book2D("HO Digi VME Occupancy Map","HO Digi VME Occupancy Map",21,-0.5,20.5,21,-0.5,20.5);
    hoHists.OCC_MAP_FIB = m_dbe->book2D("HO Digi Fiber Occupancy Map","HO Digi Fiber Occupancy Map",3,-0.5,2.5,9,-0.5,8.5);
    hoHists.OCC_MAP_DCC = m_dbe->book2D("HO Digi Spigot Occupancy Map","HO Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,0,HcalDCCHeader::SPIGOT_COUNT-1,
					fedUnpackList_.size(),0,fedUnpackList_.size());
    
}

  return;
}

void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const HFDigiCollection& hf,
				   const HcalDbService& cond){
  
  if(!m_dbe) { printf("HcalDigiMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  return; }
  
  ievt_++;
  meEVT_->Fill(ievt_);


  try{
    int nhedigi = 0;
    int nhbdigi = 0;
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      cond.makeHcalCalibration(digi.id(), &calibs_);
      if((HcalSubdetector)(digi.id().subdet())==HcalBarrel){	
	nhbdigi++;
	HcalDigiMap::fillErrors<HBHEDataFrame>(digi,hbHists.ERR_MAP_GEO,hbHists.ERR_MAP_VME,
					       hbHists.ERR_MAP_FIB,hbHists.ERR_MAP_DCC);	  
	HcalDigiMap::fillErrors<HBHEDataFrame>(digi,ERR_MAP_GEO,ERR_MAP_VME,
					       ERR_MAP_FIB,ERR_MAP_DCC);	  

	HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,hbHists.OCC_MAP_GEO1,hbHists.OCC_MAP_GEO2,
						  hbHists.OCC_MAP_GEO3,hbHists.OCC_MAP_GEO4,
						  hbHists.OCC_MAP_VME, hbHists.OCC_MAP_FIB, hbHists.OCC_MAP_DCC,
						  hbHists.OCC_ETA,hbHists.OCC_PHI,
						  occThresh_);	  
	HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,OCC_L1,OCC_L2,OCC_L3,OCC_L4,
						  OCC_ELEC_VME,OCC_ELEC_FIB,OCC_ELEC_DCC,
						  OCC_ETA,OCC_PHI,0);	  
	hbHists.DIGI_SIZE->Fill(digi.size());
	hbHists.DIGI_PRESAMPLE->Fill(digi.presamples());
	int last = -1;
	int maxa=-100;
	for (int i=0; i<digi.size(); i++)
	  if(digi.sample(i).adc()>maxa) maxa = digi.sample(i).adc();

	for (int i=0; i<digi.size(); i++) {	    
	  int capid = digi.sample(i).capid();
	  float adc = digi.sample(i).adc();
	  hbHists.SHAPE_tot->Fill(i,adc-calibs_.pedestal(capid));
	  if(maxa>occThresh_) hbHists.SHAPE_THR_tot->Fill(i,adc-calibs_.pedestal(capid));
	  hbHists.QIE_CAPID->Fill(capid);
	  hbHists.QIE_ADC->Fill(adc);
	  hbHists.QIE_CAPID->Fill(5,bitUpset(last,capid));
	  last = capid;
	  hbHists.QIE_DV->Fill(0,digi.sample(i).dv());
	  hbHists.QIE_DV->Fill(1,digi.sample(i).er());
	}    
	
	if(doPerChannel_){	  
	  vector<float> ta;
	  for (int i=0; i<digi.size(); i++) ta.push_back(digi.sample(i).adc()-digi.sample(i).capid());
	  HcalDigiPerChan::perChanHists<HBHEDataFrame>(0,digi,ta,hbHists.SHAPE,m_dbe);
	}
      }
      else if((HcalSubdetector)(digi.id().subdet())==HcalEndcap){	
	nhedigi++;
	HcalDigiMap::fillErrors<HBHEDataFrame>(digi,heHists.ERR_MAP_GEO,heHists.ERR_MAP_VME,
					       heHists.ERR_MAP_FIB,heHists.ERR_MAP_DCC);
	HcalDigiMap::fillErrors<HBHEDataFrame>(digi,ERR_MAP_GEO,ERR_MAP_VME,
					       ERR_MAP_FIB,ERR_MAP_DCC);
	
	HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,heHists.OCC_MAP_GEO1,heHists.OCC_MAP_GEO2,
						  heHists.OCC_MAP_GEO3,heHists.OCC_MAP_GEO4,
						  heHists.OCC_MAP_VME,heHists.OCC_MAP_FIB, heHists.OCC_MAP_DCC,
						  heHists.OCC_ETA,heHists.OCC_PHI,
						  occThresh_);	  
	HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,OCC_L1,OCC_L2,OCC_L3,OCC_L4,
						  OCC_ELEC_VME,OCC_ELEC_FIB,OCC_ELEC_DCC,
						  OCC_ETA,OCC_PHI,0);	  
	heHists.DIGI_SIZE->Fill(digi.size());
	heHists.DIGI_PRESAMPLE->Fill(digi.presamples());
	int last = -1;
	int maxa=-100;
	for (int i=0; i<digi.size(); i++)
	  if(digi.sample(i).adc()>maxa) maxa = digi.sample(i).adc();

	for (int i=0; i<digi.size(); i++) {	    
	  int capid = digi.sample(i).capid();
	  float adc = digi.sample(i).adc();
	  heHists.SHAPE_tot->Fill(i,adc-calibs_.pedestal(capid));
	  if(maxa>occThresh_) heHists.SHAPE_THR_tot->Fill(i,adc-calibs_.pedestal(capid));
	  heHists.QIE_CAPID->Fill(capid);
	  heHists.QIE_ADC->Fill(adc);
	  heHists.QIE_CAPID->Fill(5,bitUpset(last,capid));
	  last = capid;
	  heHists.QIE_DV->Fill(0,digi.sample(i).dv());
	  heHists.QIE_DV->Fill(1,digi.sample(i).er());
	}    
	
	if(doPerChannel_){	  
	  vector<float> ta;
	  for (int i=0; i<digi.size(); i++) ta.push_back(digi.sample(i).adc()-digi.sample(i).capid());
	  HcalDigiPerChan::perChanHists<HBHEDataFrame>(0,digi,ta,heHists.SHAPE,m_dbe);
	}
      }
    }
    
    hbHists.DIGI_NUM->Fill(nhbdigi);
    heHists.DIGI_NUM->Fill(nhedigi);
    
  } catch (...) {    
    printf("HcalDigiMonitor::processEvent  No HBHE Digis.\n");
  }
  
  try{
    hoHists.DIGI_NUM->Fill(ho.size());
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);	
      HcalDigiMap::fillErrors<HODataFrame>(digi,hoHists.ERR_MAP_GEO,hoHists.ERR_MAP_VME,
					       hoHists.ERR_MAP_FIB,hoHists.ERR_MAP_DCC);
      HcalDigiMap::fillErrors<HODataFrame>(digi,ERR_MAP_GEO,ERR_MAP_VME,
					     ERR_MAP_FIB,ERR_MAP_DCC);
      HcalDigiMap::fillOccupancy<HODataFrame>(digi,hoHists.OCC_MAP_GEO1,hoHists.OCC_MAP_GEO2,
					      hoHists.OCC_MAP_GEO3,hoHists.OCC_MAP_GEO4,
					      hoHists.OCC_MAP_VME, hoHists.OCC_MAP_FIB, hoHists.OCC_MAP_DCC,
					      hoHists.OCC_ETA,hoHists.OCC_PHI,occThresh_);	  
      HcalDigiMap::fillOccupancy<HODataFrame>(digi,OCC_L1,OCC_L2,OCC_L3,OCC_L4,
					      OCC_ELEC_VME,OCC_ELEC_FIB,OCC_ELEC_DCC,
					      OCC_ETA,OCC_PHI,0);	  
      hoHists.DIGI_SIZE->Fill(digi.size());
      hoHists.DIGI_PRESAMPLE->Fill(digi.presamples());
      int last = -1;
      int maxa=-100;
      for (int i=0; i<digi.size(); i++)
	if(digi.sample(i).adc()>maxa) maxa = digi.sample(i).adc();
      
      for (int i=0; i<digi.size(); i++) {	    
	int capid = digi.sample(i).capid();
	float adc = digi.sample(i).adc();
	hoHists.SHAPE_tot->Fill(i,adc-calibs_.pedestal(capid));
	if(maxa>occThresh_) hoHists.SHAPE_THR_tot->Fill(i,adc-calibs_.pedestal(capid));
	hoHists.QIE_CAPID->Fill(capid);
	hoHists.QIE_ADC->Fill(adc);
	hoHists.QIE_CAPID->Fill(5,bitUpset(last,capid));
	last = capid;
	hoHists.QIE_DV->Fill(0,digi.sample(i).dv());
	hoHists.QIE_DV->Fill(1,digi.sample(i).er());
      }    
      
      if(doPerChannel_){	  
	vector<float> ta;
	for (int i=0; i<digi.size(); i++) ta.push_back(digi.sample(i).adc()-digi.sample(i).capid());
	HcalDigiPerChan::perChanHists<HODataFrame>(0,digi,ta,hoHists.SHAPE,m_dbe);
      }
    }
  }
  catch (...) {
    cout << "HcalDigiMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    hfHists.DIGI_NUM->Fill(hf.size());
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);	
      HcalDigiMap::fillErrors<HFDataFrame>(digi,hfHists.ERR_MAP_GEO,hfHists.ERR_MAP_VME,
					   hfHists.ERR_MAP_FIB,hfHists.ERR_MAP_DCC);
      HcalDigiMap::fillErrors<HFDataFrame>(digi,ERR_MAP_GEO,ERR_MAP_VME,
					     ERR_MAP_FIB,ERR_MAP_DCC);
      HcalDigiMap::fillOccupancy<HFDataFrame>(digi,hfHists.OCC_MAP_GEO1,hfHists.OCC_MAP_GEO2,
					      hfHists.OCC_MAP_GEO3,hfHists.OCC_MAP_GEO4,
					      hfHists.OCC_MAP_VME, hfHists.OCC_MAP_FIB, hfHists.OCC_MAP_DCC,
					      hfHists.OCC_ETA,hfHists.OCC_PHI,
					      occThresh_);
      HcalDigiMap::fillOccupancy<HFDataFrame>(digi,OCC_L1,OCC_L2,OCC_L3,OCC_L4,
					      OCC_ELEC_VME,OCC_ELEC_FIB,OCC_ELEC_DCC,
					      OCC_ETA,OCC_PHI,0);	  
      hfHists.DIGI_SIZE->Fill(digi.size());
      hfHists.DIGI_PRESAMPLE->Fill(digi.presamples());
      int last = -1;
      int maxa=-100;
      for (int i=0; i<digi.size(); i++)
	if(digi.sample(i).adc()>maxa) maxa = digi.sample(i).adc();
      
      for (int i=0; i<digi.size(); i++) {	    
	int capid = digi.sample(i).capid();
	float adc = digi.sample(i).adc();
	hfHists.SHAPE_tot->Fill(i,adc-calibs_.pedestal(capid));
	if(maxa>occThresh_) hfHists.SHAPE_THR_tot->Fill(i,adc-calibs_.pedestal(capid));
	hfHists.QIE_CAPID->Fill(capid);
	hfHists.QIE_ADC->Fill(adc);
	hfHists.QIE_CAPID->Fill(5,bitUpset(last,capid));
	last = capid;
	hfHists.QIE_DV->Fill(0,digi.sample(i).dv());
	hfHists.QIE_DV->Fill(1,digi.sample(i).er());
      }    
      
      if(doPerChannel_){	  
	vector<float> ta;
	for (int i=0; i<digi.size(); i++) ta.push_back(digi.sample(i).adc()-digi.sample(i).capid());
	HcalDigiPerChan::perChanHists<HFDataFrame>(0,digi,ta,hfHists.SHAPE,m_dbe);
      }
    }
  } catch (...) {
    cout << "HcalDigiMonitor::processEvent  No HF Digis." << endl;
  }

  return;
}
