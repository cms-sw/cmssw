#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/HcalMonitorTasks/interface/HcalLEDMonitor.h"
#include <cmath>

HcalDigiMonitor::HcalDigiMonitor() {
  doPerChannel_ = false;
  occThresh_ = 1;
  ievt_=0;
}

HcalDigiMonitor::~HcalDigiMonitor() {}

namespace HcalDigiPerChan
{
  
  template<class Digi>
  inline void perChanHists(int id, const Digi& digi, float* ampl,std::map<HcalDetId, MonitorElement*> &tool, DQMStore* dbe, string baseFolder) {
    
    std::map<HcalDetId,MonitorElement*>::iterator _mei;

    string type;
    if (id==1) type = "HB";
    else if(id==2) type = "HE"; 
    else if(id==3) type = "HO"; 
    else if(id==4) type = "HF"; 
    
    if(dbe) dbe->setCurrentFolder(baseFolder+"/"+type);
    
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
	tool[digi.id()] =  dbe->book1D(name,name,10,-0.5,9.5); 
	for (int i=0; i<digi.size(); i++) tool[digi.id()]->Fill(i,ampl[i]);
      } // if (dbe)
    } // else
  } // inline void perChanHists

} // namespace HcalDigiPerChan



void HcalDigiMonitor::reset(){}

static bool bitUpset(int last, int now){
  if(last ==-1) return false;
  int v = last+1; if(v==4) v=0;
  if(v==now) return false;
  return true;
}

namespace HcalDigiMap{
  template<class Digi>
  inline void fillErrors(const Digi& digi, float* vals,
			 MonitorElement* mapGEO, MonitorElement* mapVME, 
			 MonitorElement* mapDCC){
    mapGEO->Fill(digi.id().ieta(),digi.id().iphi());
    float slotnum = digi.elecId().htrSlot() + 0.5*digi.elecId().htrTopBottom();
    mapVME->Fill(slotnum,digi.elecId().readoutVMECrateId());
    mapDCC->Fill(digi.elecId().spigot(),digi.elecId().dccid());
    return;
  }

  template<class Digi>
  inline void fillOccupancy(const Digi& digi, float* vals, 
			    MonitorElement* mapG1, MonitorElement* mapG2,
			    MonitorElement* mapG3, MonitorElement* mapG4,  
			    MonitorElement* mapVME, MonitorElement* mapDCC, 
			    MonitorElement* mapEta, MonitorElement* mapPhi){
    if(digi.id().depth()==1) mapG1->Fill(digi.id().ieta(),digi.id().iphi());
    else if(digi.id().depth()==2) mapG2->Fill(digi.id().ieta(),digi.id().iphi());
    else if(digi.id().depth()==3) mapG3->Fill(digi.id().ieta(),digi.id().iphi());
    else if(digi.id().depth()==4) mapG4->Fill(digi.id().ieta(),digi.id().iphi());
    float slotnum = digi.elecId().htrSlot() + 0.5*digi.elecId().htrTopBottom();
    mapVME->Fill(slotnum,digi.elecId().readoutVMECrateId());
    mapDCC->Fill(digi.elecId().spigot(),digi.elecId().dccid());
    mapEta->Fill(digi.id().ieta());
    mapPhi->Fill(digi.id().iphi());
    return;
  }

  template<class Digi>
  inline bool digiStats(const Digi& digi, HcalCalibrations calibs, float occThr, 
			float* vals,bool& err, bool& occ, bool& bitUp){
    int last = -1; float pval = -1;
    bitUp=false; err=false; occ=false;
    
    //if (digi.size()!=10) err=true; 

    for (int i=0; i<digi.size(); i++) {
      int thisCapid = digi.sample(i).capid();
      if(bitUpset(last,thisCapid)) bitUp=true;
      last = thisCapid;
      if(digi.sample(i).er()) err=true;
      if(!digi.sample(i).dv()) err=true;


      //pval = digi.sample(i).adc()-calibs.pedestal(thisCapid);

      // Occupancy was set true if ADC count above pedestal for at least one time slice
      pval = digi.sample(i).adc(); // Just want to know it's alive. //-calibs.pedestal(thisCapid);
      vals[i] = pval;
      //      if(pval>occThr) occ=true;
      if(digi.sample(i).adc()) occ=true;
    }
    if(bitUp) err=true;
    
    return err;
  }
  
}

void HcalDigiMonitor::setup(const edm::ParameterSet& ps, 
			    DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"DigiMonitor";

  occThresh_ = ps.getUntrackedParameter<int>("DigiOccThresh", -9999);
  cout << "Digi occupancy threshold set to " << occThresh_ << endl;

  doFCpeds_ = ps.getUntrackedParameter<bool>("PedestalsInFC", true);

  // Allow for diagnostic plots to be made if user wishes
  hcalHists.makeDiagnostics=ps.getUntrackedParameter<bool>("MakeDigiDiagnosticPlots",false);
  hbHists.makeDiagnostics=hcalHists.makeDiagnostics;
  heHists.makeDiagnostics=hcalHists.makeDiagnostics;
  hoHists.makeDiagnostics=hcalHists.makeDiagnostics;
  hfHists.makeDiagnostics=hcalHists.makeDiagnostics;

  if ( ps.getUntrackedParameter<bool>("DigisPerChannel", false) ) doPerChannel_ = true;  

  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 42.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -42.5);
  etaBins_ = (int)(etaMax_ - etaMin_);
  cout << "Digi eta min/max set to " << etaMin_ << "/" <<etaMax_ << endl;

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 73.5);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", -0.5);
  phiBins_ = (int)(phiMax_ - phiMin_);
  cout << "Digi phi min/max set to " << phiMin_ << "/" <<phiMax_ << endl;

   // The number of consecutive events for which a cell must not have a digi in order to be considered dead

  checkNevents_ = ps.getUntrackedParameter<int>("checkNevents",100); 

  ievt_=0;


  for (int eta=0;eta<83;++eta){
    for (int phi=0;phi<72;++phi){
      for (int depth=0;depth<4;++depth){
	pedcounts[eta][phi][depth]=0;
	rawpedsum[eta][phi][depth]=0;
	rawpedsum2[eta][phi][depth]=0;
	subpedsum[eta][phi][depth]=0;
	subpedsum2[eta][phi][depth]=0;
      }
    }
  }
  
  if ( m_dbe ) {

    m_dbe->setCurrentFolder(baseFolder_);
    meEVT_ = m_dbe->bookInt("Digi Task Event Number");    
    meEVT_->Fill(ievt_);
    
    OCC_L1 = m_dbe->book2D("Digi Depth 1 Occupancy Map","Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L1 -> setAxisTitle("ieta",1);  OCC_L1 -> setAxisTitle("iphi",2);

    OCC_L2 = m_dbe->book2D("Digi Depth 2 Occupancy Map","Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L2 -> setAxisTitle("ieta",1);  OCC_L2 -> setAxisTitle("iphi",2);

    OCC_L3 = m_dbe->book2D("Digi Depth 3 Occupancy Map","Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L3 -> setAxisTitle("ieta",1);  OCC_L3 -> setAxisTitle("iphi",2);

    OCC_L4 = m_dbe->book2D("Digi Depth 4 Occupancy Map","Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    OCC_L4 -> setAxisTitle("ieta",1);  OCC_L4 -> setAxisTitle("iphi",2);

    OCC_ETA = m_dbe->book1D("Digi Eta Occupancy Map","Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    OCC_ETA -> setAxisTitle("ieta",1);  
    OCC_ETA -> setAxisTitle("# of Events",2);

    OCC_PHI = m_dbe->book1D("Digi Phi Occupancy Map","Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);
    OCC_PHI -> setAxisTitle("iphi",1);  
    OCC_PHI -> setAxisTitle("# of Events",2);

    OCC_ELEC_VME = m_dbe->book2D("Digi VME Occupancy Map","Digi VME Occupancy Map",40,-0.25,19.75,18,-0.5,17.5);
    OCC_ELEC_VME -> setAxisTitle("HTR Slot",1);  
    OCC_ELEC_VME -> setAxisTitle("VME Crate Id",2);

    OCC_ELEC_DCC = m_dbe->book2D("Digi Spigot Occupancy Map","Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    OCC_ELEC_DCC -> setAxisTitle("Spigot",1);  
    OCC_ELEC_DCC -> setAxisTitle("DCC Id",2);

    ERR_MAP_GEO = m_dbe->book2D("Digi Geo Error Map","Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    ERR_MAP_GEO -> setAxisTitle("ieta",1);  
    ERR_MAP_GEO -> setAxisTitle("iphi",2);

    ERR_MAP_VME = m_dbe->book2D("Digi VME Error Map","Digi VME Error Map",40,-0.25,19.75,18,-0.5,17.5);
    ERR_MAP_VME -> setAxisTitle("HTR Slot",1);  
    ERR_MAP_VME -> setAxisTitle("VME Crate Id",2);

    ERR_MAP_DCC = m_dbe->book2D("Digi Spigot Error Map","Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    ERR_MAP_DCC -> setAxisTitle("Spigot",1);  
    ERR_MAP_DCC -> setAxisTitle("DCC Id",2);

    CAPID_T0 = m_dbe->book1D("Capid 1st Time Slice","Capid for 1st Time Slice",7,-3.5,3.5);
    CAPID_T0 -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
    CAPID_T0 -> setAxisTitle("# of Events",2);
 
    DIGI_NUM = m_dbe->book1D("# of Digis","# of Digis",910,-0.5,9099.5);
    DIGI_NUM -> setAxisTitle("# of Digis",1);  
    DIGI_NUM -> setAxisTitle("# of Events",2);

    BQDIGI_NUM = m_dbe->book1D("# Bad Qual Digis","# Bad Qual Digis",910,-0.5,9099.5);
    BQDIGI_NUM -> setAxisTitle("# Bad Quality Digis",1);  
    BQDIGI_NUM -> setAxisTitle("# of Events",2);

    BQDIGI_FRAC =  m_dbe->book1D("Bad Digi Fraction","Bad Digi Fraction",220,-0.05,1.05);
    BQDIGI_FRAC -> setAxisTitle("Bad Quality Digi Fraction",1);  
    BQDIGI_FRAC -> setAxisTitle("# of Events",2);


    // HB Plots
    m_dbe->setCurrentFolder(baseFolder_+"/HB");

    hbHists.check=ps.getUntrackedParameter<bool>("checkHB","true");
    hbHists.SHAPE_tot =  m_dbe->book1D("HB Digi Shape","HB Digi Shape",10,-0.5,9.5);
    hbHists.SHAPE_THR_tot =  m_dbe->book1D("HB Digi Shape - over thresh","HB Digi Shape - over thresh",10,-0.5,9.5);

    hbHists.DIGI_NUM =  m_dbe->book1D("HB # of Digis","HB # of Digis",2700,-0.5,2699.5);
    hbHists.DIGI_NUM -> setAxisTitle("# of Digis",1);  
    hbHists.DIGI_NUM -> setAxisTitle("# of Events",2);

    hbHists.BQDIGI_NUM =  m_dbe->book1D("HB # Bad Qual Digis","HB # Bad Qual Digis",2700,-0.5,2699.5);
    hbHists.BQDIGI_NUM -> setAxisTitle("# Bad Quality Digis",1);  
    hbHists.BQDIGI_NUM -> setAxisTitle("# of Events",2);

    hbHists.BQDIGI_FRAC =  m_dbe->book1D("HB Bad Digi Fraction","HB Bad Digi Fraction",220,-0.05,1.05);
    hbHists.BQDIGI_FRAC -> setAxisTitle("Bad Quality Digi Fraction",1);  
    hbHists.BQDIGI_FRAC -> setAxisTitle("# of Events",2);

    hbHists.CAPID_T0 = m_dbe->book1D("HB Capid 1st Time Slice","HB Capid for 1st Time Slice",7,-3.5,3.5);
    hbHists.CAPID_T0 -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
    hbHists.CAPID_T0 -> setAxisTitle("# of Events",2);

    hbHists.DIGI_SIZE =  m_dbe->book1D("HB Digi Size","HB Digi Size",50,0,50);
    hbHists.DIGI_PRESAMPLE =  m_dbe->book1D("HB Digi Presamples","HB Digi Presamples",50,0,50);
    hbHists.QIE_CAPID =  m_dbe->book1D("HB QIE Cap-ID","HB QIE Cap-ID",6,-0.5,5.5);
    hbHists.QIE_ADC = m_dbe->book1D("HB QIE ADC Value","HB QIE ADC Value",100,0,200);
    hbHists.QIE_DV = m_dbe->book1D("HB QIE Data Valid Err Bits","HB QIE Data Valid Err Bits",4,-0.5,3.5);
    hbHists.QIE_DV ->setBinLabel(1,"Err=0, DV=0",1);
    hbHists.QIE_DV ->setBinLabel(2,"Err=0, DV=1",1);
    hbHists.QIE_DV ->setBinLabel(3,"Err=1, DV=0",1);
    hbHists.QIE_DV ->setBinLabel(4,"Err=1, DV=1",1);
    
    hbHists.ERR_MAP_GEO = m_dbe->book2D("HB Digi Geo Error Map","HB Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.ERR_MAP_GEO -> setAxisTitle("ieta",1);  
    hbHists.ERR_MAP_GEO -> setAxisTitle("iphi",2);


    hbHists.ERR_MAP_VME = m_dbe->book2D("HB Digi VME Error Map","HB Digi VME Error Map",40,-0.25,19.75,18,-0.5,17.5);
    hbHists.ERR_MAP_VME -> setAxisTitle("HTR Slot",1);  
    hbHists.ERR_MAP_VME -> setAxisTitle("VME Crate Id",2);


    hbHists.ERR_MAP_DCC = m_dbe->book2D("HB Digi Spigot Error Map","HB Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    hbHists.ERR_MAP_DCC -> setAxisTitle("Spigot",1);  
    hbHists.ERR_MAP_DCC -> setAxisTitle("DCC Id",2);

    hbHists.OCC_MAP_GEO1 = m_dbe->book2D("HB Digi Depth 1 Occupancy Map","HB Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO1 -> setAxisTitle("ieta",1);  
    hbHists.OCC_MAP_GEO1 -> setAxisTitle("iphi",2);

    hbHists.OCC_MAP_GEO2 = m_dbe->book2D("HB Digi Depth 2 Occupancy Map","HB Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO2 -> setAxisTitle("ieta",1);  
    hbHists.OCC_MAP_GEO2 -> setAxisTitle("iphi",2);

    hbHists.OCC_MAP_GEO3 = m_dbe->book2D("HB Digi Depth 3 Occupancy Map","HB Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO3 -> setAxisTitle("ieta",1);  
    hbHists.OCC_MAP_GEO3 -> setAxisTitle("iphi",2);

    hbHists.OCC_MAP_GEO4 = m_dbe->book2D("HB Digi Depth 4 Occupancy Map","HB Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.OCC_MAP_GEO4 -> setAxisTitle("ieta",1);  
    hbHists.OCC_MAP_GEO4 -> setAxisTitle("iphi",2);

    hbHists.OCC_ETA = m_dbe->book1D("HB Digi Eta Occupancy Map","HB Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    hbHists.OCC_ETA -> setAxisTitle("ieta",1);  
    hbHists.OCC_ETA -> setAxisTitle("# of Events",2);

    hbHists.OCC_PHI = m_dbe->book1D("HB Digi Phi Occupancy Map","HB Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);
    hbHists.OCC_PHI -> setAxisTitle("iphi",1);  
    hbHists.OCC_PHI -> setAxisTitle("# of Events",2);

    hbHists.OCC_MAP_VME = m_dbe->book2D("HB Digi VME Occupancy Map","HB Digi VME Occupancy Map",40,-0.25,19.75,18,-0.5,17.5);
    hbHists.OCC_MAP_VME -> setAxisTitle("HTR Slot",1);  
    hbHists.OCC_MAP_VME -> setAxisTitle("VME Crate Id",2);

    hbHists.OCC_MAP_DCC = m_dbe->book2D("HB Digi Spigot Occupancy Map","HB Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    hbHists.OCC_MAP_DCC -> setAxisTitle("Spigot",1);  
    hbHists.OCC_MAP_DCC -> setAxisTitle("DCC Id",2);

    hbHists.TS_SUM_P.push_back( m_dbe->book1D("HBP Timeslices 2 and 3", "HBP Timeslices 2 and 3", 50, -5, 45) );
    hbHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hbHists.TS_SUM_P.push_back( m_dbe->book1D("HBP Timeslices 3 and 4", "HBP Timeslices 3 and 4", 50, -5, 45) );
    hbHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hbHists.TS_SUM_P.push_back( m_dbe->book1D("HBP Timeslices 4 and 5", "HBP Timeslices 4 and 5", 50, -5, 45) );
    hbHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hbHists.TS_SUM_M.push_back( m_dbe->book1D("HBM Timeslices 2 and 3", "HBM Timeslices 2 and 3", 50, -5, 45) );
    hbHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    hbHists.TS_SUM_M.push_back( m_dbe->book1D("HBM Timeslices 3 and 4", "HBM Timeslices 3 and 4", 50, -5, 45) );
    hbHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    hbHists.TS_SUM_M.push_back( m_dbe->book1D("HBM Timeslices 4 and 5", "HBM Timeslices 4 and 5", 50, -5, 45) );
    hbHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);

    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    heHists.check=ps.getUntrackedParameter<bool>("checkHE","true");
    heHists.SHAPE_tot =  m_dbe->book1D("HE Digi Shape","HE Digi Shape",10,-0.5,9.5);
    heHists.SHAPE_THR_tot =  m_dbe->book1D("HE Digi Shape - over thresh","HE Digi Shape - over thresh",10,-0.5,9.5);
    heHists.DIGI_NUM =  m_dbe->book1D("HE # of Digis","HE # of Digis",2700,-0.5,2699.5);
    heHists.DIGI_NUM -> setAxisTitle("# of Digis",1);  
    heHists.DIGI_NUM -> setAxisTitle("# of Events",2);

    heHists.DIGI_SIZE =  m_dbe->book1D("HE Digi Size","HE Digi Size",50,0,50);
    heHists.DIGI_PRESAMPLE =  m_dbe->book1D("HE Digi Presamples","HE Digi Presamples",50,0,50);
    heHists.QIE_CAPID =  m_dbe->book1D("HE QIE Cap-ID","HE QIE Cap-ID",6,-0.5,5.5);
    heHists.QIE_ADC = m_dbe->book1D("HE QIE ADC Value","HE QIE ADC Value",100,0,200);
    heHists.QIE_DV = m_dbe->book1D("HE QIE Data Valid Err Bits","HE QIE Data Valid Err Bits",4,-0.5,3.5);
    heHists.QIE_DV ->setBinLabel(1,"Err=0, DV=0",1);
    heHists.QIE_DV ->setBinLabel(2,"Err=0, DV=1",1);
    heHists.QIE_DV ->setBinLabel(3,"Err=1, DV=0",1);
    heHists.QIE_DV ->setBinLabel(4,"Err=1, DV=1",1);

    heHists.BQDIGI_NUM =  m_dbe->book1D("HE # Bad Qual Digis","HE # Bad Qual Digis",2700,-0.5,2699.5);
    heHists.BQDIGI_NUM -> setAxisTitle("# Bad Quality Digis",1);  
    heHists.BQDIGI_NUM -> setAxisTitle("# of Events",2);

    heHists.BQDIGI_FRAC =  m_dbe->book1D("HE Bad Digi Fraction","HE Bad Digi Fraction",220,-0.05,1.05);
    heHists.BQDIGI_FRAC -> setAxisTitle("Bad Quality Digi Fraction",1);  
    heHists.BQDIGI_FRAC -> setAxisTitle("# of Events",2);

    heHists.CAPID_T0 = m_dbe->book1D("HE Capid 1st Time Slice","HE Capid for 1st Time Slice",7,-3.5,3.5);
    heHists.CAPID_T0 -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
    heHists.CAPID_T0 -> setAxisTitle("# of Events",2);

    heHists.ERR_MAP_GEO = m_dbe->book2D("HE Digi Geo Error Map","HE Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.ERR_MAP_GEO -> setAxisTitle("ieta",1);  
    heHists.ERR_MAP_GEO -> setAxisTitle("iphi",2);

    heHists.ERR_MAP_VME = m_dbe->book2D("HE Digi VME Error Map","HE Digi VME Error Map",40,-0.25,19.75,18,-0.5,17.5);
    heHists.ERR_MAP_VME -> setAxisTitle("HTR Slot",1);  
    heHists.ERR_MAP_VME -> setAxisTitle("VME Crate Id",2);

    heHists.ERR_MAP_DCC = m_dbe->book2D("HE Digi Spigot Error Map","HE Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    heHists.ERR_MAP_DCC -> setAxisTitle("Spigot",1);  
    heHists.ERR_MAP_DCC -> setAxisTitle("DCC Id",2);

    heHists.OCC_MAP_GEO1 = m_dbe->book2D("HE Digi Depth 1 Occupancy Map","HE Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO1 -> setAxisTitle("ieta",1);  
    heHists.OCC_MAP_GEO1 -> setAxisTitle("iphi",2);

    heHists.OCC_MAP_GEO2 = m_dbe->book2D("HE Digi Depth 2 Occupancy Map","HE Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO2 -> setAxisTitle("ieta",1);  
    heHists.OCC_MAP_GEO2 -> setAxisTitle("iphi",2);

    heHists.OCC_MAP_GEO3 = m_dbe->book2D("HE Digi Depth 3 Occupancy Map","HE Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO3 -> setAxisTitle("ieta",1);  
    heHists.OCC_MAP_GEO3 -> setAxisTitle("iphi",2);

    heHists.OCC_MAP_GEO4 = m_dbe->book2D("HE Digi Depth 4 Occupancy Map","HE Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.OCC_MAP_GEO4 -> setAxisTitle("ieta",1);  
    heHists.OCC_MAP_GEO4 -> setAxisTitle("iphi",2);

    heHists.OCC_ETA = m_dbe->book1D("HE Digi Eta Occupancy Map","HE Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    heHists.OCC_ETA -> setAxisTitle("ieta",1);  
    heHists.OCC_ETA -> setAxisTitle("# of Events",2);

    heHists.OCC_PHI = m_dbe->book1D("HE Digi Phi Occupancy Map","HE Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);
    heHists.OCC_PHI -> setAxisTitle("iphi",1);  
    heHists.OCC_PHI -> setAxisTitle("# of Events",2);

    heHists.OCC_MAP_VME = m_dbe->book2D("HE Digi VME Occupancy Map","HE Digi VME Occupancy Map",40,-0.25,19.75,18,-0.5,17.5);
    heHists.OCC_MAP_VME -> setAxisTitle("HTR Slot",1);  
    heHists.OCC_MAP_VME -> setAxisTitle("VME Crate Id",2);

    heHists.OCC_MAP_DCC = m_dbe->book2D("HE Digi Spigot Occupancy Map","HE Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    heHists.OCC_MAP_DCC -> setAxisTitle("Spigot",1);  
    heHists.OCC_MAP_DCC -> setAxisTitle("DCC Id",2);

    heHists.TS_SUM_P.push_back( m_dbe->book1D("HEP Timeslices 2 and 3", "HEP Timeslices 2 and 3", 50, -5, 45) );
    heHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    heHists.TS_SUM_P.push_back( m_dbe->book1D("HEP Timeslices 3 and 4", "HEP Timeslices 3 and 4", 50, -5, 45) );
    heHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    heHists.TS_SUM_P.push_back( m_dbe->book1D("HEP Timeslices 4 and 5", "HEP Timeslices 4 and 5", 50, -5, 45) );
    heHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    heHists.TS_SUM_M.push_back( m_dbe->book1D("HEM Timeslices 2 and 3", "HEM Timeslices 2 and 3", 50, -5, 45) );
    heHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    heHists.TS_SUM_M.push_back( m_dbe->book1D("HEM Timeslices 3 and 4", "HEM Timeslices 3 and 4", 50, -5, 45) );
    heHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    heHists.TS_SUM_M.push_back( m_dbe->book1D("HEM Timeslices 4 and 5", "HEM Timeslices 4 and 5", 50, -5, 45) );
    heHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);

    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    hfHists.check=ps.getUntrackedParameter<bool>("checkHF","true");
    hfHists.SHAPE_tot =  m_dbe->book1D("HF Digi Shape","HF Digi Shape",10,-0.5,9.5);
    hfHists.SHAPE_THR_tot =  m_dbe->book1D("HF Digi Shape - over thresh","HF Digi Shape - over thresh",10,-0.5,9.5);
    hfHists.DIGI_NUM =  m_dbe->book1D("HF # of Digis","HF # of Digis",1800,-0.5,1799.5);
    hfHists.DIGI_NUM -> setAxisTitle("# of Digis",1);  
    hfHists.DIGI_NUM -> setAxisTitle("# of Events",2);

    hfHists.DIGI_SIZE =  m_dbe->book1D("HF Digi Size","HF Digi Size",50,0,50);
    hfHists.DIGI_PRESAMPLE =  m_dbe->book1D("HF Digi Presamples","HF Digi Presamples",50,0,50);
    hfHists.QIE_CAPID =  m_dbe->book1D("HF QIE Cap-ID","HF QIE Cap-ID",6,-0.5,5.5);
    hfHists.QIE_ADC = m_dbe->book1D("HF QIE ADC Value","HF QIE ADC Value",100,0,200);
    hfHists.QIE_DV = m_dbe->book1D("HF QIE Data Valid Err Bits","HF QIE Data Valid Err Bits",4,-0.5,3.5);
    hfHists.QIE_DV ->setBinLabel(1,"Err=0, DV=0",1);
    hfHists.QIE_DV ->setBinLabel(2,"Err=0, DV=1",1);
    hfHists.QIE_DV ->setBinLabel(3,"Err=1, DV=0",1);
    hfHists.QIE_DV ->setBinLabel(4,"Err=1, DV=1",1);

    hfHists.BQDIGI_NUM =  m_dbe->book1D("HF # Bad Qual Digis","HF # Bad Qual Digis",1800,-0.5,1799.5);
    hfHists.BQDIGI_NUM -> setAxisTitle("# Bad Quality Digis",1);  
    hfHists.BQDIGI_NUM -> setAxisTitle("# of Events",2);

    hfHists.BQDIGI_FRAC =  m_dbe->book1D("HF Bad Digi Fraction","HF Bad Digi Fraction",220,-0.05,1.05);
    hfHists.BQDIGI_FRAC -> setAxisTitle("Bad Quality Digi Fraction",1);  
    hfHists.BQDIGI_FRAC -> setAxisTitle("# of Events",2);

    hfHists.CAPID_T0 = m_dbe->book1D("HF Capid 1st Time Slice","HF Capid for 1st Time Slice",7,-3.5,3.5);
    hfHists.CAPID_T0 -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
    hfHists.CAPID_T0 -> setAxisTitle("# of Events",2);

    hfHists.ERR_MAP_GEO = m_dbe->book2D("HF Digi Geo Error Map","HF Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.ERR_MAP_GEO -> setAxisTitle("ieta",1);  
    hfHists.ERR_MAP_GEO -> setAxisTitle("iphi",2);

    hfHists.ERR_MAP_VME = m_dbe->book2D("HF Digi VME Error Map","HF Digi VME Error Map",40,-0.25,19.75,18,-0.5,17.5);
    hfHists.ERR_MAP_VME -> setAxisTitle("HTR Slot",1);  
    hfHists.ERR_MAP_VME -> setAxisTitle("VME Crate Id",2);

    hfHists.ERR_MAP_DCC = m_dbe->book2D("HF Digi Spigot Error Map","HF Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    hfHists.ERR_MAP_DCC -> setAxisTitle("Spigot",1);  
    hfHists.ERR_MAP_DCC -> setAxisTitle("DCC Id",2);

    hfHists.OCC_MAP_GEO1 = m_dbe->book2D("HF Digi Depth 1 Occupancy Map","HF Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO1 -> setAxisTitle("ieta",1);  
    hfHists.OCC_MAP_GEO1 -> setAxisTitle("iphi",2);

    hfHists.OCC_MAP_GEO2 = m_dbe->book2D("HF Digi Depth 2 Occupancy Map","HF Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO2 -> setAxisTitle("ieta",1);  
    hfHists.OCC_MAP_GEO2 -> setAxisTitle("iphi",2);

    hfHists.OCC_MAP_GEO3 = m_dbe->book2D("HF Digi Depth 3 Occupancy Map","HF Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO3 -> setAxisTitle("ieta",1);  
    hfHists.OCC_MAP_GEO3 -> setAxisTitle("iphi",2);

    hfHists.OCC_MAP_GEO4 = m_dbe->book2D("HF Digi Depth 4 Occupancy Map","HF Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.OCC_MAP_GEO4 -> setAxisTitle("ieta",1);  
    hfHists.OCC_MAP_GEO4 -> setAxisTitle("iphi",2);

    hfHists.OCC_ETA = m_dbe->book1D("HF Digi Eta Occupancy Map","HF Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    hfHists.OCC_ETA -> setAxisTitle("ieta",1);  
    hfHists.OCC_ETA -> setAxisTitle("# of Events",2);

    hfHists.OCC_PHI = m_dbe->book1D("HF Digi Phi Occupancy Map","HF Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);
    hfHists.OCC_PHI -> setAxisTitle("iphi",1);  
    hfHists.OCC_PHI -> setAxisTitle("# of Events",2);


    hfHists.OCC_MAP_VME = m_dbe->book2D("HF Digi VME Occupancy Map","HF Digi VME Occupancy Map",40,-0.25,19.75,18,-0.5,17.5);
    hfHists.OCC_MAP_VME -> setAxisTitle("HTR Slot",1);  
    hfHists.OCC_MAP_VME -> setAxisTitle("VME Crate Id",2);


    hfHists.OCC_MAP_DCC = m_dbe->book2D("HF Digi Spigot Occupancy Map","HF Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    hfHists.OCC_MAP_DCC -> setAxisTitle("Spigot",1);  
    hfHists.OCC_MAP_DCC -> setAxisTitle("DCC Id",2);

    hfHists.TS_SUM_P.push_back( m_dbe->book1D("HFP Timeslices 2 and 3", "HFP Timeslices 2 and 3", 50, -5, 45) );
    hfHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hfHists.TS_SUM_P.push_back( m_dbe->book1D("HFP Timeslices 3 and 4", "HFP Timeslices 3 and 4", 50, -5, 45) );
    hfHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hfHists.TS_SUM_P.push_back( m_dbe->book1D("HFP Timeslices 4 and 5", "HFP Timeslices 4 and 5", 50, -5, 45) );
    hfHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hfHists.TS_SUM_M.push_back( m_dbe->book1D("HFM Timeslices 2 and 3", "HFM Timeslices 2 and 3", 50, -5, 45) );
    hfHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    hfHists.TS_SUM_M.push_back( m_dbe->book1D("HFM Timeslices 3 and 4", "HFM Timeslices 3 and 4", 50, -5, 45) );
    hfHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    hfHists.TS_SUM_M.push_back( m_dbe->book1D("HFM Timeslices 4 and 5", "HFM Timeslices 4 and 5", 50, -5, 45) );
    hfHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);

    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    hoHists.check=ps.getUntrackedParameter<bool>("checkHO","true");
    hoHists.SHAPE_tot =  m_dbe->book1D("HO Digi Shape","HO Digi Shape",10,-0.5,9.5);
    hoHists.SHAPE_THR_tot =  m_dbe->book1D("HO Digi Shape - over thresh","HO Digi Shape - over thresh",10,-0.5,9.5);
    hoHists.DIGI_NUM =  m_dbe->book1D("HO # of Digis","HO # of Digis",2200,-0.5,2199.5);
    hoHists.DIGI_NUM -> setAxisTitle("# of Digis",1);  
    hoHists.DIGI_NUM -> setAxisTitle("# of Events",2);

    hoHists.DIGI_SIZE =  m_dbe->book1D("HO Digi Size","HO Digi Size",50,0,50);
    hoHists.DIGI_PRESAMPLE =  m_dbe->book1D("HO Digi Presamples","HO Digi Presamples",50,0,50);
    hoHists.QIE_CAPID =  m_dbe->book1D("HO QIE Cap-ID","HO QIE Cap-ID",6,-0.5,5.5);
    hoHists.QIE_ADC = m_dbe->book1D("HO QIE ADC Value","HO QIE ADC Value",100,0,200);
    hoHists.QIE_DV = m_dbe->book1D("HO QIE Data Valid Err Bits","HO QIE Data Valid Err Bits",4,-0.5,3.5);
    hoHists.QIE_DV ->setBinLabel(1,"Err=0, DV=0",1);
    hoHists.QIE_DV ->setBinLabel(2,"Err=0, DV=1",1);
    hoHists.QIE_DV ->setBinLabel(3,"Err=1, DV=0",1);
    hoHists.QIE_DV ->setBinLabel(4,"Err=1, DV=1",1);

    hoHists.BQDIGI_NUM =  m_dbe->book1D("HO # Bad Qual Digis","HO # Bad Qual Digis",2200,-0.5,2199.5);
    hoHists.BQDIGI_NUM -> setAxisTitle("# Bad Quality Digis",1);  
    hoHists.BQDIGI_NUM -> setAxisTitle("# of Events",2);

    hoHists.BQDIGI_FRAC =  m_dbe->book1D("HO Bad Digi Fraction","HO Bad Digi Fraction",220,-0.05,1.05);
    hoHists.BQDIGI_FRAC -> setAxisTitle("Bad Quality Digi Fraction",1);  
    hoHists.BQDIGI_FRAC -> setAxisTitle("# of Events",2);

    hoHists.CAPID_T0 = m_dbe->book1D("HO Capid 1st Time Slice","HO Capid for 1st Time Slice",7,-3.5,3.5);
    hoHists.CAPID_T0 -> setAxisTitle("CapID (T0) - 1st CapId (T0)",1);  
    hoHists.CAPID_T0 -> setAxisTitle("# of Events",2);

    hoHists.ERR_MAP_GEO = m_dbe->book2D("HO Digi Geo Error Map","HO Digi Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.ERR_MAP_GEO -> setAxisTitle("ieta",1);  
    hoHists.ERR_MAP_GEO -> setAxisTitle("iphi",2);

    hoHists.ERR_MAP_VME = m_dbe->book2D("HO Digi VME Error Map","HO Digi VME Error Map",40,-0.25,19.75,18,-0.5,17.5);
    hoHists.ERR_MAP_VME -> setAxisTitle("HTR Slot",1);  
    hoHists.ERR_MAP_VME -> setAxisTitle("VME Crate Id",2);

    hoHists.ERR_MAP_DCC = m_dbe->book2D("HO Digi Spigot Error Map","HO Digi Spigot Error Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    hoHists.ERR_MAP_DCC -> setAxisTitle("Spigot",1);  
    hoHists.ERR_MAP_DCC -> setAxisTitle("DCC Id",2);

    hoHists.OCC_MAP_GEO1 = m_dbe->book2D("HO Digi Depth 1 Occupancy Map","HO Digi Depth 1 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO1 -> setAxisTitle("ieta",1);  
    hoHists.OCC_MAP_GEO1 -> setAxisTitle("iphi",2);

    hoHists.OCC_MAP_GEO2 = m_dbe->book2D("HO Digi Depth 2 Occupancy Map","HO Digi Depth 2 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO2 -> setAxisTitle("ieta",1);  
    hoHists.OCC_MAP_GEO2 -> setAxisTitle("iphi",2);

    hoHists.OCC_MAP_GEO3 = m_dbe->book2D("HO Digi Depth 3 Occupancy Map","HO Digi Depth 3 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO3 -> setAxisTitle("ieta",1);  
    hoHists.OCC_MAP_GEO3 -> setAxisTitle("iphi",2);

    hoHists.OCC_MAP_GEO4 = m_dbe->book2D("HO Digi Depth 4 Occupancy Map","HO Digi Depth 4 Occupancy Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.OCC_MAP_GEO4 -> setAxisTitle("ieta",1);  
    hoHists.OCC_MAP_GEO4 -> setAxisTitle("iphi",2);

    hoHists.OCC_ETA = m_dbe->book1D("HO Digi Eta Occupancy Map","HO Digi Eta Occupancy Map",etaBins_,etaMin_,etaMax_);
    hoHists.OCC_ETA -> setAxisTitle("ieta",1);  
    hoHists.OCC_ETA -> setAxisTitle("# of Events",2);

    hoHists.OCC_PHI = m_dbe->book1D("HO Digi Phi Occupancy Map","HO Digi Phi Occupancy Map",phiBins_,phiMin_,phiMax_);
    hoHists.OCC_PHI -> setAxisTitle("iphi",1);  
    hoHists.OCC_PHI -> setAxisTitle("# of Events",2);


    hoHists.OCC_MAP_VME = m_dbe->book2D("HO Digi VME Occupancy Map","HO Digi VME Occupancy Map",40,-0.25,19.75,18,-0.5,17.5);
    hoHists.OCC_MAP_VME -> setAxisTitle("HTR Slot",1);  
    hoHists.OCC_MAP_VME -> setAxisTitle("VME Crate Id",2);

    hoHists.OCC_MAP_DCC = m_dbe->book2D("HO Digi Spigot Occupancy Map","HO Digi Spigot Occupancy Map",
					HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
					36,-0.5,35.5);
    hoHists.OCC_MAP_DCC -> setAxisTitle("Spigot",1);  
    hoHists.OCC_MAP_DCC -> setAxisTitle("DCC Id",2);

    hoHists.TS_SUM_P.push_back( m_dbe->book1D("HOP Timeslices 2 and 3", "HOP Timeslices 2 and 3", 50, -5, 45) );
    hoHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hoHists.TS_SUM_P.push_back( m_dbe->book1D("HOP Timeslices 3 and 4", "HOP Timeslices 3 and 4", 50, -5, 45) );
    hoHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hoHists.TS_SUM_P.push_back( m_dbe->book1D("HOP Timeslices 4 and 5", "HOP Timeslices 4 and 5", 50, -5, 45) );
    hoHists.TS_SUM_P.back()->setAxisTitle("Sum of ADC counts", 1);
    hoHists.TS_SUM_M.push_back( m_dbe->book1D("HOM Timeslices 2 and 3", "HOM Timeslices 2 and 3", 50, -5, 45) );
    hoHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    hoHists.TS_SUM_M.push_back( m_dbe->book1D("HOM Timeslices 3 and 4", "HOM Timeslices 3 and 4", 50, -5, 45) );
    hoHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);
    hoHists.TS_SUM_M.push_back( m_dbe->book1D("HOM Timeslices 4 and 5", "HOM Timeslices 4 and 5", 50, -5, 45) );
    hoHists.TS_SUM_M.back()->setAxisTitle("Sum of ADC counts", 1);

    // Summary histograms for storing problem info (cells with either a digi error or with low digi occupancy)
    m_dbe->setCurrentFolder(baseFolder_+"/HCAL");

    hcalHists.check=true;
    hcalHists.PROBLEMDIGICELLS=m_dbe->book2D("HCALProblemDigiCells", 
					  "HCAL Bad Digi rate for potentially bad cells", 
					  etaBins_, etaMin_, etaMax_, 
					  phiBins_, phiMin_, phiMax_); 

    RAW_PEDESTAL_MEAN[0] = m_dbe->book2D("RawPedestalMeanDepth1","Raw Pedestal Mean Value Map (Time Slices 0-1) Depth 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_MEAN[0]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_MEAN[0]->setAxisTitle("#iphi",2);
    RAW_PEDESTAL_RMS[0]  = m_dbe->book2D("RawPedestalRMSDepth1", "Raw Pedestal RMS Map (Time Slices 0-1) Depth 1", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_RMS[0]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_RMS[0]->setAxisTitle("#iphi",2);
    
    SUB_PEDESTAL_MEAN[0] = m_dbe->book2D("SubPedestalMeanDepth1","Sub Pedestal Mean Value Map (Time Slices 0-1) Depth 1",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_MEAN[0]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_MEAN[0]->setAxisTitle("#iphi",2);
    SUB_PEDESTAL_RMS[0]  = m_dbe->book2D("SubPedestalRMSDepth1", "Sub Pedestal RMS Map (Time Slices 0-1) Depth 1", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_RMS[0]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_RMS[0]->setAxisTitle("#iphi",2);
    
    RAW_PEDESTAL_MEAN[1] = m_dbe->book2D("RawPedestalMeanDepth2","Raw Pedestal Mean Value Map (Time Slices 0-1) Depth 2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_MEAN[1]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_MEAN[1]->setAxisTitle("#iphi",2);
    RAW_PEDESTAL_RMS[1]  = m_dbe->book2D("RawPedestalRMSDepth2", "Raw Pedestal RMS Map (Time Slices 0-1) Depth 2", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_RMS[1]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_RMS[1]->setAxisTitle("#iphi",2);


    SUB_PEDESTAL_MEAN[1] = m_dbe->book2D("SubPedestalMeanDepth2","Sub Pedestal Mean Value Map (Time Slices 0-1) Depth 2",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_MEAN[1]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_MEAN[1]->setAxisTitle("#iphi",2);
    SUB_PEDESTAL_RMS[1]  = m_dbe->book2D("SubPedestalRMSDepth2", "Sub Pedestal RMS Map (Time Slices 0-1) Depth 2", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_RMS[1]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_RMS[1]->setAxisTitle("#iphi",2);

    RAW_PEDESTAL_MEAN[2] = m_dbe->book2D("RawPedestalMeanDepth3","Raw Pedestal Mean Value Map (Time Slices 0-1) Depth 3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_MEAN[2]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_MEAN[2]->setAxisTitle("#iphi",2);
    RAW_PEDESTAL_RMS[2]  = m_dbe->book2D("RawPedestalRMSDepth3", "Raw Pedestal RMS Map (Time Slices 0-1) Depth 3", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_RMS[2]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_RMS[2]->setAxisTitle("#iphi",2);
    
    SUB_PEDESTAL_MEAN[2] = m_dbe->book2D("SubPedestalMeanDepth3","Sub Pedestal Mean Value Map (Time Slices 0-1) Depth 3",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_MEAN[2]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_MEAN[2]->setAxisTitle("#iphi",2);
    SUB_PEDESTAL_RMS[2]  = m_dbe->book2D("SubPedestalRMSDepth3", "Sub Pedestal RMS Map (Time Slices 0-1) Depth 3", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_RMS[2]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_RMS[2]->setAxisTitle("#iphi",2);

    RAW_PEDESTAL_MEAN[3] = m_dbe->book2D("RawPedestalMeanDepth4","Raw Pedestal Mean Value Map (Time Slices 0-1) Depth 4",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_MEAN[3]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_MEAN[3]->setAxisTitle("#iphi",2);
    RAW_PEDESTAL_RMS[3]  = m_dbe->book2D("RawPedestalRMSDepth4", "Raw Pedestal RMS Map (Time Slices 0-1) Depth 4", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    RAW_PEDESTAL_RMS[3]->setAxisTitle("#ieta",1);
    RAW_PEDESTAL_RMS[3]->setAxisTitle("#iphi",2);
    
    SUB_PEDESTAL_MEAN[3] = m_dbe->book2D("SubPedestalMeanDepth4","Sub Pedestal Mean Value Map (Time Slices 0-1) Depth 4",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_MEAN[3]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_MEAN[3]->setAxisTitle("#iphi",2);
    SUB_PEDESTAL_RMS[3]  = m_dbe->book2D("SubPedestalRMSDepth4", "Sub Pedestal RMS Map (Time Slices 0-1) Depth 4", etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    SUB_PEDESTAL_RMS[3]->setAxisTitle("#ieta",1);
    SUB_PEDESTAL_RMS[3]->setAxisTitle("#iphi",2);

    std::stringstream histname; 
    std::stringstream histtitle; 

    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HCAL/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"HCALProblemDigiCells_depth"<<d+1; 
	histtitle<<"HCAL Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hcalHists.PROBLEMDIGICELLS_DEPTH.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(), 
								 etaBins_,etaMin_,etaMax_, 
								 phiBins_,phiMin_,phiMax_)); 
      }

    //HB 
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    hbHists.PROBLEMDIGICELLS=m_dbe->book2D("HBProblemDigiCells", 
					  "HB Bad Digi rate for potentially bad cells", 
					  etaBins_, etaMin_, etaMax_, 
					  phiBins_, phiMin_, phiMax_); 
    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HB/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"HBProblemDigiCells_depth"<<d+1; 
	histtitle<<"HB Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hbHists.PROBLEMDIGICELLS_DEPTH.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(), 
							       etaBins_,etaMin_,etaMax_, 
							       phiBins_,phiMin_,phiMax_)); 
      }

   //HE 
    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    heHists.PROBLEMDIGICELLS=m_dbe->book2D("HEProblemDigiCells", 
					  "HE Bad Digi rate for potentially bad cells", 
					  etaBins_, etaMin_, etaMax_, 
					  phiBins_, phiMin_, phiMax_); 
    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HE/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"HEProblemDigiCells_depth"<<d+1; 
	histtitle<<"HE Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	heHists.PROBLEMDIGICELLS_DEPTH.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(), 
							       etaBins_,etaMin_,etaMax_, 
							       phiBins_,phiMin_,phiMax_)); 
      }

    //HO 
    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    hoHists.PROBLEMDIGICELLS=m_dbe->book2D("HOProblemDigiCells", 
					  "HO Bad Digi rate for potentially bad cells", 
					  etaBins_, etaMin_, etaMax_, 
					  phiBins_, phiMin_, phiMax_); 
    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HO/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"HOProblemDigiCells_depth"<<d+1; 
	histtitle<<"HO Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hoHists.PROBLEMDIGICELLS_DEPTH.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(), 
							       etaBins_,etaMin_,etaMax_, 
							       phiBins_,phiMin_,phiMax_)); 
      }

   //HF 
    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    hfHists.PROBLEMDIGICELLS=m_dbe->book2D("HFProblemDigiCells", 
					  "HF Bad Digi rate for potentially bad cells", 
					  etaBins_, etaMin_, etaMax_, 
					  phiBins_, phiMin_, phiMax_); 
    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HF/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"HFProblemDigiCells_depth"<<d+1; 
	histtitle<<"HF Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hfHists.PROBLEMDIGICELLS_DEPTH.push_back(m_dbe->book2D(histname.str().c_str(),histtitle.str().c_str(), 
							       etaBins_,etaMin_,etaMax_, 
							       phiBins_,phiMin_,phiMax_)); 
      }


    // Form temp histograms
    // HCAL
    hcalHists.PROBLEMDIGICELLS_TEMP=new TH2F("HCALProblemDigiCells_temp", 
					"HCAL Bad Digi rate for potentially bad cells", 
					etaBins_, etaMin_, etaMax_, 
					phiBins_, phiMin_, phiMax_); 

    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HCAL/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"tempHCALProblemDigiCells_depth"<<d+1; 
	histtitle<<"HCAL Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH.push_back(new TH2F(histname.str().c_str(),histtitle.str().c_str(), 
								 etaBins_,etaMin_,etaMax_, 
								 phiBins_,phiMin_,phiMax_)); 
      }

    // HB
    hbHists.PROBLEMDIGICELLS_TEMP=new TH2F("HBProblemDigiCells_temp", 
					"HB Bad Digi rate for potentially bad cells", 
					etaBins_, etaMin_, etaMax_, 
					phiBins_, phiMin_, phiMax_); 

    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HB/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"tempHBProblemDigiCells_depth"<<d+1; 
	histtitle<<"HB Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hbHists.PROBLEMDIGICELLS_TEMP_DEPTH.push_back(new TH2F(histname.str().c_str(),histtitle.str().c_str(), 
								      etaBins_,etaMin_,etaMax_, 
								      phiBins_,phiMin_,phiMax_)); 
      }

    // HE
    heHists.PROBLEMDIGICELLS_TEMP=new TH2F("HEProblemDigiCells_temp", 
					"HE Bad Digi rate for potentially bad cells", 
					etaBins_, etaMin_, etaMax_, 
					phiBins_, phiMin_, phiMax_); 

    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HE/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"tempHEProblemDigiCells_depth"<<d+1; 
	histtitle<<"HE Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	heHists.PROBLEMDIGICELLS_TEMP_DEPTH.push_back(new TH2F(histname.str().c_str(),histtitle.str().c_str(), 
								      etaBins_,etaMin_,etaMax_, 
								      phiBins_,phiMin_,phiMax_)); 
      }

    // HO
    hoHists.PROBLEMDIGICELLS_TEMP=new TH2F("HOProblemDigiCells_temp", 
					"HO Bad Digi rate for potentially bad cells", 
					etaBins_, etaMin_, etaMax_, 
					phiBins_, phiMin_, phiMax_); 

    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HO/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"tempHOProblemDigiCells_depth"<<d+1; 
	histtitle<<"HO Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hoHists.PROBLEMDIGICELLS_TEMP_DEPTH.push_back(new TH2F(histname.str().c_str(),histtitle.str().c_str(), 
								      etaBins_,etaMin_,etaMax_, 
								      phiBins_,phiMin_,phiMax_)); 
      }

    // HF
    hfHists.PROBLEMDIGICELLS_TEMP=new TH2F("HFProblemDigiCells_temp", 
					"HF Bad Digi rate for potentially bad cells", 
					etaBins_, etaMin_, etaMax_, 
					phiBins_, phiMin_, phiMax_); 

    for (int d=0;d<4;++d)
      {
	m_dbe->setCurrentFolder(baseFolder_+"/HF/expertPlots"); 
	histname.str(""); 
	histtitle.str(""); 
	histname<<"tempHFProblemDigiCells_depth"<<d+1; 
	histtitle<<"HF Bad Digi rate for potentially bad cells (depth "<<d+1<<")"; 
	hfHists.PROBLEMDIGICELLS_TEMP_DEPTH.push_back(new TH2F(histname.str().c_str(),histtitle.str().c_str(), 
								      etaBins_,etaMin_,etaMax_, 
								      phiBins_,phiMin_,phiMax_)); 
      }


    hcalHists.PROBLEMDIGICELLS -> setAxisTitle("ieta",1);  
    hcalHists.PROBLEMDIGICELLS -> setAxisTitle("iphi",2);
    hbHists.PROBLEMDIGICELLS -> setAxisTitle("ieta",1);  
    hbHists.PROBLEMDIGICELLS -> setAxisTitle("iphi",2);
    heHists.PROBLEMDIGICELLS -> setAxisTitle("ieta",1);  
    heHists.PROBLEMDIGICELLS -> setAxisTitle("iphi",2);
    hoHists.PROBLEMDIGICELLS -> setAxisTitle("ieta",1);  
    hoHists.PROBLEMDIGICELLS -> setAxisTitle("iphi",2);
    hfHists.PROBLEMDIGICELLS -> setAxisTitle("ieta",1);  
    hfHists.PROBLEMDIGICELLS -> setAxisTitle("iphi",2);
    
    for (int d=0;d<4;++d)
      {
	hcalHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("ieta",1);  
	hcalHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("iphi",2);
	hbHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("ieta",1);  
	hbHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("iphi",2);
	heHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("ieta",1);  
	heHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("iphi",2);
	hoHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("ieta",1);  
	hoHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("iphi",2);
	hfHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("ieta",1);  
	hfHists.PROBLEMDIGICELLS_DEPTH[d] -> setAxisTitle("iphi",2);
      }

  
  }// if (m_dbe)

  return;
}

void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const HFDigiCollection& hf,
				   const HcalDbService& cond,
				   const HcalUnpackerReport& report)
{
  
  if(!m_dbe) { 
    if(fVerbosity) printf("HcalDigiMonitor::processEvent   DQMStore not instantiated!!!\n");  
    return; 
  }
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  //EmptyDigiFill(); // fill all cells as being empty; "unfill" for all digis that are found -- not used at the moment


  int nbqdigi_report = report.badQualityDigis();
  if (nbqdigi_report != 0)BQDIGI_NUM->Fill(nbqdigi_report);

  float normVals[10]; bool digiErr=false;
  bool digiOcc=false; bool digiUpset=false;
  int ndigi = 0;  int nbqdigi = 0;

  CaloSamples tool;
  const HcalQIECoder* channelCoder;
  const HcalQIEShape* shape;

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
	  
  try{
    int nhedigi = 0;   int nhbdigi = 0;
    int nhbbqdigi = 0;  int nhebqdigi = 0;
    int firsthbcap = -1; int firsthecap = -1;
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); ++j){
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      

      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private; we will need it for forming subtracted pedestals
      int iEta = digi.id().ieta();
      int iPhi = digi.id().iphi();
      int iDepth = digi.id().depth();

      HcalDigiMap::digiStats(digi, calibs_, occThresh_, normVals, digiErr, digiOcc, digiUpset); 

      if((HcalSubdetector)(digi.id().subdet())==HcalBarrel){	
	if (!hbHists.check) continue;

	nhbdigi++;  ndigi++;
	// Digi found; "unfill it" so that it doesn't appear empty:
	if ( digiOcc) // require digi to have a value (don't know if this is the best way to proceed?)
	  {
	    hbHists.PROBLEMDIGICELLS_TEMP->Fill(iEta,iPhi,1);
	    hbHists.PROBLEMDIGICELLS_TEMP_DEPTH[iDepth-1]->Fill(iEta,iPhi,1);
	
	    hcalHists.PROBLEMDIGICELLS_TEMP->Fill(iEta,iPhi,1);
	    hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[iDepth-1]->Fill(iEta,iPhi,1);
	  }
	if(digiErr){
	  nhbbqdigi++; nbqdigi++;

	  hbHists.PROBLEMDIGICELLS->Fill(iEta,iPhi);
	  hbHists.PROBLEMDIGICELLS_DEPTH[iDepth-1]->Fill(iEta,iPhi);
	  hcalHists.PROBLEMDIGICELLS->Fill(iEta,iPhi);
	  hcalHists.PROBLEMDIGICELLS_DEPTH[iDepth-1]->Fill(iEta,iPhi);

	  HcalDigiMap::fillErrors<HBHEDataFrame>(digi,normVals,
						 hbHists.ERR_MAP_GEO,hbHists.ERR_MAP_VME,
						 hbHists.ERR_MAP_DCC);	  
	  
	  HcalDigiMap::fillErrors<HBHEDataFrame>(digi,normVals,
						 ERR_MAP_GEO,ERR_MAP_VME,
						 ERR_MAP_DCC);	  
	}

	if(digiOcc){
	  HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,normVals,
						    hbHists.OCC_MAP_GEO1,hbHists.OCC_MAP_GEO2,
						    hbHists.OCC_MAP_GEO3,hbHists.OCC_MAP_GEO4,
						    hbHists.OCC_MAP_VME, 
						    hbHists.OCC_MAP_DCC,
						    hbHists.OCC_ETA,hbHists.OCC_PHI);
	  
	  HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,normVals,
						    OCC_L1,OCC_L2,OCC_L3,OCC_L4,
						    OCC_ELEC_VME,OCC_ELEC_DCC,
						    OCC_ETA,OCC_PHI);	  
	}

	hbHists.DIGI_SIZE->Fill(digi.size());
	hbHists.DIGI_PRESAMPLE->Fill(digi.presamples());

	if (firsthbcap == -1) firsthbcap = digi.sample(0).capid();
	int capdif = digi.sample(0).capid() - firsthbcap;
	capdif = capdif%3 - capdif/3;
	hbHists.CAPID_T0->Fill(capdif);
	CAPID_T0->Fill(capdif);

	//for timing plot, find max-TS
	int maxadc=0;
	float myval=0;

	
	if (doFCpeds_)
	  {
	    channelCoder = cond.getHcalCoder(digi.id());
	    HcalCoderDb coderDB(*channelCoder, *shape);
	    coderDB.adc2fC(digi,tool);
	    // digi (ADC) is input, tool (fC) is output
	  }

	for (int j=0; j<digi.size(); ++j)
	  {     
	    if (digi.sample(j).adc() > maxadc) maxadc = digi.sample(j).adc();
	    // add pedestal plots
	    if (j<2) // only plot for first 2 time slices
	      {	      
		pedcounts[iEta+41][iPhi-1][iDepth-1]++;
		myval=digi.sample(j).adc();
		rawpedsum[iEta+41][iPhi-1][iDepth-1]+=myval;
		rawpedsum2[iEta+41][iPhi-1][iDepth-1]+=myval*myval;
		

		if (doFCpeds_) // Pedestals in fC; convert digi ADC to fC as well
		  myval=tool[j]-calibs_.pedestal(digi.sample(j).capid());
		else
		  myval=digi.sample(j).adc()-calibs_.pedestal(digi.sample(j).capid());
		subpedsum[iEta+41][iPhi-1][iDepth-1]+=myval;
		subpedsum2[iEta+41][iPhi-1][iDepth-1]+=myval*myval;
	      }
	}

	for (int i=0; i<digi.size(); ++i) {	    
	  hbHists.QIE_CAPID->Fill(digi.sample(i).capid());
	  hbHists.QIE_ADC->Fill(digi.sample(i).adc());
	  //Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	  //   hbHists.SHAPE_tot->Fill(i,normVals[i]);
	  int jadc=digi.sample(i).adc();
	  float tmp = (LedMonAdc2fc[jadc]+0.5);
	  hbHists.SHAPE_tot->Fill(i,tmp);
	  
	  //Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	  //and introducing threshold able to find muons
	  //   if(digiOcc) hbHists.SHAPE_THR_tot->Fill(i,normVals[i]);
	  if(maxadc>10) hbHists.SHAPE_THR_tot->Fill(i,tmp);	  
	  if(digiUpset) hbHists.QIE_CAPID->Fill(5);
	  int dver = 2*digi.sample(i).er() + digi.sample(i).dv();
	  hbHists.QIE_DV->Fill(dver);
	}    
	
	if(doPerChannel_)	  
	    HcalDigiPerChan::perChanHists<HBHEDataFrame>(1,digi,normVals,hbHists.SHAPE,m_dbe,baseFolder_);


	if (iEta > 0) {
	  hbHists.TS_SUM_P[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	  hbHists.TS_SUM_P[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	  hbHists.TS_SUM_P[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
	}
	else if (iEta < 0) {
	  hbHists.TS_SUM_M[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	  hbHists.TS_SUM_M[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	  hbHists.TS_SUM_M[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
	}
      }
      else if((HcalSubdetector)(digi.id().subdet())==HcalEndcap){
	if (!heHists.check) continue;
	nhedigi++;  ndigi++;

 	if ( digiOcc)
	  {
	    heHists.PROBLEMDIGICELLS_TEMP->Fill(iEta,iPhi,1);
	    heHists.PROBLEMDIGICELLS_TEMP_DEPTH[iDepth-1]->Fill(iEta,iPhi,1);
	    hcalHists.PROBLEMDIGICELLS_TEMP->Fill(iEta,iPhi,1);
	    hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[iDepth-1]->Fill(iEta,iPhi,1);
	  }

	if(digiErr){
	  nhebqdigi++; nbqdigi++;
	  heHists.PROBLEMDIGICELLS->Fill(iEta,iPhi);
	  heHists.PROBLEMDIGICELLS_DEPTH[iDepth-1]->Fill(iEta,iPhi);
	  hcalHists.PROBLEMDIGICELLS->Fill(iEta,iPhi);
	  hcalHists.PROBLEMDIGICELLS_DEPTH[iDepth-1]->Fill(iEta,iPhi);
	  HcalDigiMap::fillErrors<HBHEDataFrame>(digi,normVals,
						 heHists.ERR_MAP_GEO,heHists.ERR_MAP_VME,
						 heHists.ERR_MAP_DCC);	  

	  HcalDigiMap::fillErrors<HBHEDataFrame>(digi,normVals,
						 ERR_MAP_GEO,ERR_MAP_VME,
						 ERR_MAP_DCC);	  
	}

	if(digiOcc){
	  HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,normVals,
						    heHists.OCC_MAP_GEO1,heHists.OCC_MAP_GEO2,
						    heHists.OCC_MAP_GEO3,heHists.OCC_MAP_GEO4,
						    heHists.OCC_MAP_VME, 
						    heHists.OCC_MAP_DCC,
						    heHists.OCC_ETA,heHists.OCC_PHI);
	  
	  HcalDigiMap::fillOccupancy<HBHEDataFrame>(digi,normVals,
						    OCC_L1,OCC_L2,OCC_L3,OCC_L4,
						    OCC_ELEC_VME,OCC_ELEC_DCC,
						    OCC_ETA,OCC_PHI);	  
	}
	
	heHists.DIGI_SIZE->Fill(digi.size());
	heHists.DIGI_PRESAMPLE->Fill(digi.presamples());

	if (firsthecap == -1) firsthecap = digi.sample(0).capid();
	int capdif = digi.sample(0).capid() - firsthecap;
	capdif = capdif%3 - capdif/3;
	heHists.CAPID_T0->Fill(capdif);
	CAPID_T0->Fill(capdif);


	//for timing plot, find max-TS
	int maxadc=0;
	float myval=0;

	if (doFCpeds_)
	  {
	    channelCoder = cond.getHcalCoder(digi.id());
	    HcalCoderDb coderDB(*channelCoder, *shape);
	    coderDB.adc2fC(digi,tool);
	    // digi (ADC) is input, tool (fC) is output
	  }


	for (int j=0; j<digi.size(); ++j)
	  {     
	    if (digi.sample(j).adc() > maxadc) maxadc = digi.sample(j).adc();
	    if (j<2) // only plot for first 2 time slices
	      {	      
		pedcounts[iEta+41][iPhi-1][iDepth-1]++;
		myval=digi.sample(j).adc();
		rawpedsum[iEta+41][iPhi-1][iDepth-1]+=myval;
		rawpedsum2[iEta+41][iPhi-1][iDepth-1]+=myval*myval;
		if (doFCpeds_) // Pedestals in fC; convert digi ADC to fC as well
		  myval=tool[j]-calibs_.pedestal(digi.sample(j).capid());
		else
		  myval=digi.sample(j).adc()-calibs_.pedestal(digi.sample(j).capid());
		subpedsum[iEta+41][iPhi-1][iDepth-1]+=myval;
		subpedsum2[iEta+41][iPhi-1][iDepth-1]+=myval*myval;
	      }
	  }

	for (int i=0; i<digi.size(); ++i) {	    
	  heHists.QIE_CAPID->Fill(digi.sample(i).capid());
	  heHists.QIE_ADC->Fill(digi.sample(i).adc());
	  //Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	  //   heHists.SHAPE_tot->Fill(i,normVals[i]);
	  int jadc=digi.sample(i).adc();
	  float tmp = (LedMonAdc2fc[jadc]+0.5);
	  heHists.SHAPE_tot->Fill(i,tmp);
	  
	  //Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	  //and introducing threshold able to find muons
	  //   if(digiOcc) heHists.SHAPE_THR_tot->Fill(i,normVals[i]);
	  if(maxadc>10) heHists.SHAPE_THR_tot->Fill(i,tmp);
	  if(digiUpset) heHists.QIE_CAPID->Fill(5);
	  int dver = 2*digi.sample(i).er() + digi.sample(i).dv();
	  heHists.QIE_DV->Fill(dver);
	}    
	
	if(doPerChannel_)
	  HcalDigiPerChan::perChanHists<HBHEDataFrame>(2,digi,normVals,heHists.SHAPE,m_dbe,baseFolder_);


	if (iEta > 0) {
	  heHists.TS_SUM_P[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	  heHists.TS_SUM_P[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	  heHists.TS_SUM_P[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
	}
	else if (iEta < 0) {
	  heHists.TS_SUM_M[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	  heHists.TS_SUM_M[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	  heHists.TS_SUM_M[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
	}

      }
    }
    
    hbHists.DIGI_NUM->Fill(nhbdigi);
    hbHists.BQDIGI_NUM->Fill(nhbbqdigi);
    if (nhbdigi != 0)hbHists.BQDIGI_FRAC->Fill((1.0*nhbbqdigi)/(1.0*nhbdigi));

    heHists.DIGI_NUM->Fill(nhedigi);
    heHists.BQDIGI_NUM->Fill(nhebqdigi);
    if (nhedigi != 0)heHists.BQDIGI_FRAC->Fill((1.0*nhebqdigi)/(1.0*nhedigi));
    
  } catch (...) {    
    if(fVerbosity) printf("HcalDigiMonitor::processEvent  No HBHE Digis.\n");
  }

  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalDigiMonitor DIGI HBHE -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
	  
  try{
    int firsthocap = -1; int nhobqdigi = 0;

    int nhodigi = ho.size();
    //    hoHists.DIGI_NUM->Fill(ho.size());


    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); ++j){
      if (!hoHists.check) continue;
      const HODataFrame digi = (const HODataFrame)(*j);	
      
      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private; we will need it for forming subtracted pedestals
      int iEta = digi.id().ieta();
      int iPhi = digi.id().iphi();
      int iDepth = digi.id().depth();
      HcalDigiMap::digiStats(digi, calibs_, occThresh_, normVals, digiErr, digiOcc, digiUpset);     
      
      if ( digiOcc)
	{
	  hoHists.PROBLEMDIGICELLS_TEMP->Fill(digi.id().ieta(),digi.id().iphi(),1);
	  hoHists.PROBLEMDIGICELLS_TEMP_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi(),1);
	  hcalHists.PROBLEMDIGICELLS_TEMP->Fill(digi.id().ieta(),digi.id().iphi(),1);
	  hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi(),1);
	}
      if(digiErr){
	hoHists.PROBLEMDIGICELLS->Fill(digi.id().ieta(),digi.id().iphi());
	hoHists.PROBLEMDIGICELLS_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi());
	hcalHists.PROBLEMDIGICELLS->Fill(digi.id().ieta(),digi.id().iphi());
	hcalHists.PROBLEMDIGICELLS_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi());

	nhobqdigi++; nbqdigi++;
      }
      
      if (digiErr){
	HcalDigiMap::fillErrors<HODataFrame>(digi,normVals,
					       hoHists.ERR_MAP_GEO,hoHists.ERR_MAP_VME,
					       hoHists.ERR_MAP_DCC);	  
	
	HcalDigiMap::fillErrors<HODataFrame>(digi,normVals,
					       ERR_MAP_GEO,ERR_MAP_VME,
					       ERR_MAP_DCC);	  
      }
      
      if(digiOcc){
	HcalDigiMap::fillOccupancy<HODataFrame>(digi,normVals,
						  hoHists.OCC_MAP_GEO1,hoHists.OCC_MAP_GEO2,
						  hoHists.OCC_MAP_GEO3,hoHists.OCC_MAP_GEO4,
						hoHists.OCC_MAP_VME, 
						  hoHists.OCC_MAP_DCC,
						  hoHists.OCC_ETA,hoHists.OCC_PHI);
	
	HcalDigiMap::fillOccupancy<HODataFrame>(digi,normVals,
						  OCC_L1,OCC_L2,OCC_L3,OCC_L4,
						  OCC_ELEC_VME,OCC_ELEC_DCC,
						  OCC_ETA,OCC_PHI);	  
      }
      

      hoHists.DIGI_SIZE->Fill(digi.size());
      hoHists.DIGI_PRESAMPLE->Fill(digi.presamples());

      if (firsthocap == -1) firsthocap = digi.sample(0).capid();
      int capdif = digi.sample(0).capid() - firsthocap;
      capdif = capdif%3 - capdif/3;
      hoHists.CAPID_T0->Fill(capdif);
      CAPID_T0->Fill(capdif);
      
      //for timing plot, find max-TS
      int maxadc=0;
      float myval=0;
      
      if (doFCpeds_)
	{
	  channelCoder = cond.getHcalCoder(digi.id());
	  HcalCoderDb coderDB(*channelCoder, *shape);
	  coderDB.adc2fC(digi,tool);
	  // digi (ADC) is input, tool (fC) is output
	}
      
      for (int j=0; j<digi.size(); ++j)
	{     
	  if (digi.sample(j).adc() > maxadc) maxadc = digi.sample(j).adc();

	  if (j<2) // only plot for first 2 time slices
	    {	      
	      pedcounts[iEta+41][iPhi-1][iDepth-1]++;
	      myval=digi.sample(j).adc();
	      rawpedsum[iEta+41][iPhi-1][iDepth-1]+=myval;
	      rawpedsum2[iEta+41][iPhi-1][iDepth-1]+=myval*myval;
	      if (doFCpeds_) // Pedestals in fC; convert digi ADC to fC as well
		myval=tool[j]-calibs_.pedestal(digi.sample(j).capid());
	      else
		myval=digi.sample(j).adc()-calibs_.pedestal(digi.sample(j).capid());
	      subpedsum[iEta+41][iPhi-1][iDepth-1]+=myval;
	      subpedsum2[iEta+41][iPhi-1][iDepth-1]+=myval*myval;
	    }
	}

      for (int i=0; i<digi.size(); ++i) {	    
	hoHists.QIE_CAPID->Fill(digi.sample(i).capid());
	hoHists.QIE_ADC->Fill(digi.sample(i).adc());
	//Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	//  hoHists.SHAPE_tot->Fill(i,normVals[i]);

	int jadc=digi.sample(i).adc();
	float tmp = (LedMonAdc2fc[jadc]+0.5);

	hoHists.SHAPE_tot->Fill(i,tmp);
	
	//Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	//and introducing threshold able to find muons
	//   if(digiOcc) hoHists.SHAPE_THR_tot->Fill(i,normVals[i]);
	if(maxadc>10) hoHists.SHAPE_THR_tot->Fill(i,tmp);
	if(digiUpset) hoHists.QIE_CAPID->Fill(5);
	int dver = 2*digi.sample(i).er() + digi.sample(i).dv();
	hoHists.QIE_DV->Fill(dver);
      } // for (int i=0; i<digi.size();++i)


      if(doPerChannel_)	  
	HcalDigiPerChan::perChanHists<HODataFrame>(3,digi,normVals,hoHists.SHAPE,m_dbe, baseFolder_);


	if (digi.id().ieta() > 0) {
	  hoHists.TS_SUM_P[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	  hoHists.TS_SUM_P[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	  hoHists.TS_SUM_P[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
	}
	else if (digi.id().ieta() < 0) {
	  hoHists.TS_SUM_M[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	  hoHists.TS_SUM_M[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	  hoHists.TS_SUM_M[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
	}

    }
    hoHists.DIGI_NUM->Fill(nhodigi);
    hoHists.BQDIGI_NUM->Fill(nhobqdigi);
    if (nhodigi != 0)hoHists.BQDIGI_FRAC->Fill((1.0*nhobqdigi)/(1.0*nhodigi));
    ndigi = ndigi + nhodigi;
  }
  catch (...) {
    if(fVerbosity) cout << "HcalDigiMonitor::processEvent  No HO Digis." << endl;
  }

  if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalDigiMonitor DIGI HO -> "<<cpu_timer.cpuTime()<<endl;
      cpu_timer.reset(); cpu_timer.start();
    }
  
  try{
    int firsthfcap = -1; int nhfbqdigi = 0;
    int nhfdigi = hf.size();
    //    hfHists.DIGI_NUM->Fill(hf.size());
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); ++j){
      if (!hfHists.check) continue;
      const HFDataFrame digi = (const HFDataFrame)(*j);
 
      if ( digiOcc)
	{
	  hfHists.PROBLEMDIGICELLS_TEMP->Fill(digi.id().ieta(),digi.id().iphi(),1);
	  hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi(),1);
	  hcalHists.PROBLEMDIGICELLS_TEMP->Fill(digi.id().ieta(),digi.id().iphi(),1);
	  hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi(),1);
	}

      calibs_= cond.getHcalCalibrations(digi.id());  // Old method was made private; we will need it for forming subtracted pedestals
      int iEta = digi.id().ieta();
      int iPhi = digi.id().iphi();
      int iDepth = digi.id().depth();
      HcalDigiMap::digiStats(digi, calibs_, occThresh_, normVals, digiErr, digiOcc, digiUpset);      
	
      if(digiErr){
	hfHists.PROBLEMDIGICELLS->Fill(digi.id().ieta(),digi.id().iphi());
	hfHists.PROBLEMDIGICELLS_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi());
	hcalHists.PROBLEMDIGICELLS->Fill(digi.id().ieta(),digi.id().iphi());
	hcalHists.PROBLEMDIGICELLS_DEPTH[digi.id().depth()-1]->Fill(digi.id().ieta(),digi.id().iphi());
	nhfbqdigi++;  nbqdigi++;
	HcalDigiMap::fillErrors<HFDataFrame>(digi,normVals,
					       hfHists.ERR_MAP_GEO,hfHists.ERR_MAP_VME,
					       hfHists.ERR_MAP_DCC);	  
	
	HcalDigiMap::fillErrors<HFDataFrame>(digi,normVals,
					       ERR_MAP_GEO,ERR_MAP_VME,
					       ERR_MAP_DCC);	  
      }
      
      if(digiOcc){
	HcalDigiMap::fillOccupancy<HFDataFrame>(digi,normVals,
						  hfHists.OCC_MAP_GEO1,hfHists.OCC_MAP_GEO2,
						  hfHists.OCC_MAP_GEO3,hfHists.OCC_MAP_GEO4,
						  hfHists.OCC_MAP_VME, 
						  hfHists.OCC_MAP_DCC,
						  hfHists.OCC_ETA,hfHists.OCC_PHI);
	
	HcalDigiMap::fillOccupancy<HFDataFrame>(digi,normVals,
						  OCC_L1,OCC_L2,OCC_L3,OCC_L4,
						  OCC_ELEC_VME,OCC_ELEC_DCC,
						  OCC_ETA,OCC_PHI);	  
      }
      
      hfHists.DIGI_SIZE->Fill(digi.size());
      hfHists.DIGI_PRESAMPLE->Fill(digi.presamples());

      if (firsthfcap == -1) firsthfcap = digi.sample(0).capid();
      int capdif = digi.sample(0).capid() - firsthfcap;
      capdif = capdif%3 - capdif/3;
      hfHists.CAPID_T0->Fill(capdif);
      CAPID_T0->Fill(capdif);
      
      //for timing plot, find max-TS
      int maxadc=0;
      float myval=0;

      if (doFCpeds_)
	{
	  channelCoder = cond.getHcalCoder(digi.id());
	  HcalCoderDb coderDB(*channelCoder, *shape);
	  coderDB.adc2fC(digi,tool);
	  // digi (ADC) is input, tool (fC) is output
	}

      for (int j=0; j<digi.size(); ++j)
	{     
	  if (digi.sample(j).adc() > maxadc) maxadc = digi.sample(j).adc();

	  if (j<2) // only plot for first 2 time slices
	    {	      
	      // Depth values increased by 2 to avoid overlap with HE at |eta|=29
	      pedcounts[iEta+41][iPhi-1][iDepth+1]++;
	      myval=digi.sample(j).adc();
	      rawpedsum[iEta+41][iPhi-1][iDepth+1]+=myval;
	      rawpedsum2[iEta+41][iPhi-1][iDepth+1]+=myval*myval;
	      if (doFCpeds_) // Pedestals in fC; convert digi ADC to fC as well
		myval=tool[j]-calibs_.pedestal(digi.sample(j).capid());
	      else
		myval=digi.sample(j).adc()-calibs_.pedestal(digi.sample(j).capid());
	      subpedsum[iEta+41][iPhi-1][iDepth+1]+=myval;
	      subpedsum2[iEta+41][iPhi-1][iDepth+1]+=myval*myval;
	    }
	}
   
      for (int i=0; i<digi.size(); ++i) {	    
	hfHists.QIE_CAPID->Fill(digi.sample(i).capid());
	hfHists.QIE_ADC->Fill(digi.sample(i).adc());
	//Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	//  hfHists.SHAPE_tot->Fill(i,normVals[i]);
	int jadc=digi.sample(i).adc();
	float tmp = (LedMonAdc2fc[jadc]+0.5);
	hfHists.SHAPE_tot->Fill(i,tmp);
	
	//Timing plot: skipping ped. subtraction and fC conversion, just lin.adc counts
	//and introducing threshold able to find muons
	//  if(digiOcc) hfHists.SHAPE_THR_tot->Fill(i,normVals[i]);
        if(maxadc>10) hfHists.SHAPE_THR_tot->Fill(i,tmp);
	if(digiUpset) hfHists.QIE_CAPID->Fill(5);
	int dver = 2*digi.sample(i).er() + digi.sample(i).dv();
	hfHists.QIE_DV->Fill(dver);
      }    


      if(doPerChannel_)	  
	HcalDigiPerChan::perChanHists<HFDataFrame>(4,digi,normVals,hfHists.SHAPE,m_dbe, baseFolder_);


      if (digi.id().ieta() > 0) {
	hfHists.TS_SUM_P[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	hfHists.TS_SUM_P[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	hfHists.TS_SUM_P[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
      }
      else if (digi.id().ieta() < 0) {
	hfHists.TS_SUM_M[0]->Fill(digi.sample(2).adc() + digi.sample(3).adc());
	hfHists.TS_SUM_M[1]->Fill(digi.sample(3).adc() + digi.sample(4).adc());
	hfHists.TS_SUM_M[2]->Fill(digi.sample(4).adc() + digi.sample(5).adc());	  
      }

    }
    hfHists.DIGI_NUM->Fill(nhfdigi);
    hfHists.BQDIGI_NUM->Fill(nhfbqdigi);
    if (nhfdigi != 0)hfHists.BQDIGI_FRAC->Fill((1.0*nhfbqdigi)/(1.0*nhfdigi));
    ndigi = ndigi + nhfdigi;
  } catch (...) {
    if(fVerbosity) cout << "HcalDigiMonitor::processEvent  No HF Digis." << endl;
  }

 if (showTiming)
    {
      cpu_timer.stop();
      cout <<"TIMER:: HcalDigiMonitor DIGI HF -> "<<cpu_timer.cpuTime()<<endl;
    }

  DIGI_NUM->Fill(ndigi);
  if (nbqdigi !=0) BQDIGI_NUM->Fill(nbqdigi);
  if (ndigi !=0){
    double bqfrac;
    if (nbqdigi == 0){
      bqfrac = (1.0*nbqdigi_report)/(1.0*(ndigi+nbqdigi_report));
      //      cout << "Bad Digis from report, bqfrac:  " << nbqdigi_report << "  "<< bqfrac<< endl; 
    }
    else{
      bqfrac = (1.0*nbqdigi)/(1.0*ndigi);
      //      cout << "Bad Digis counted, bqfrac:  " << nbqdigi << "  "<< bqfrac<< endl;
    }
    BQDIGI_FRAC->Fill(bqfrac);
  }

  // Check for consistently missing digis
  if (ievt_>0 && ievt_%(checkNevents_)==0)
    {
      if (showTiming) 
	{
	  cpu_timer.reset(); cpu_timer.start();
	}
      reset_Nevents();
      if (showTiming)
	{
	  cpu_timer.stop();
	  cout <<"TIMER:: HcalDigiMonitor DIGI CheckNevents -> "<<cpu_timer.cpuTime()<<endl;
	}
    }

  return;
} // void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,...)


void HcalDigiMonitor::EmptyDigiFill()
{
  int eta, phi;

  // Fill all digi values with -1 every event
  // Digis that are found will then fill a 1, cancelling this out
  // (so that histogram should be filled with 0's if all digis are in every event)

  // In the case of no digis seen, this will
  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1; iphi<=phiBins_;++iphi)
	{
	  eta=ieta+int(etaMin_)-1;
	  phi=iphi+int(phiMin_)-1;

	  // ignore unphysical values
	  if (phi==0) continue; 
	  if (phi>72) continue;
	  if (eta==0) continue; 
	  if (abs(eta)>41) continue;

	  // HB first
	  if (abs(eta)<17)
	    {
	      hbHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hbHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
	    
	  
	      if (abs(eta)>14) // last two rows of HB have two depths
		{
		  hbHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  hbHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);
		}
	    } // if (abs(eta)<17)  // HB Block

	  // HO loop -- depth = 4
	  if (abs(eta)<16)
	    {
	      hoHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hoHists.PROBLEMDIGICELLS_TEMP_DEPTH[3]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[3]->Fill(eta,phi,-1);
	    } // if (abs(eta)<16)
	  
	  // HE loop (careful; phi values are odd only for eta>20)

	  if (abs(eta)==16) // at eta=16, HE depth=3
	    {
	      heHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      heHists.PROBLEMDIGICELLS_TEMP_DEPTH[2]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[2]->Fill(eta,phi,-1);
	    }

	  if (abs(eta)>16 && abs(eta)<30) // HE has depth = 1-2 for eta=18-29; 27-28 also depth=3
	    // This differs from documentation, which claimed that depth=3 for eta=28-29
	    {
	      if (abs(eta)<21 ||(abs(eta)>20 && (phi%2)==1)) // decreased phi segementation above eta=20
		{
		  heHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  heHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
		}  
	      if (abs(eta)>17) // only one layer for HE in eta=17 -- skip it when filling depth=2
		{
		  if (abs(eta)>20 && (phi%2)==0)
		    continue;
		  heHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  heHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);
		}

	      if (abs(eta)>26 && abs(eta)<29 && (phi%2)==1) // depth 3
		{
		  heHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  heHists.PROBLEMDIGICELLS_TEMP_DEPTH[2]->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
		  hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[2]->Fill(eta,phi,-1);
		} // if (abs(eta)>26 && abs(eta)<29 ...)
	    } //  if (abs(eta)>15 && abs(eta)<30) // ends HE loop
	  


	  // HF Loop
	  if (abs(eta)>28 && abs(eta)<40 && (phi%2)==1)
	    {
	      //depth1
	      hfHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);

	      //depth2
	      hfHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);

	    }
	  else if (abs(eta)>39 && (phi%4)==3)
	    {
	      //depth1
	      hfHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->Fill(eta,phi,-1);
	      
	      //depth2
	      hfHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP->Fill(eta,phi,-1);
	      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->Fill(eta,phi,-1);

	    } // end HF loop
	  
	} //for (int iphi=1; ...)
    } // for (int ieta=1; ...)
} // void HcalDigiMonitor::EmptyDigiFill()


void HcalDigiMonitor::reset_Nevents(void)
{
  
  if (fVerbosity)
    cout <<"<HcalDigiMonitor> Entered reset_Nevents routine"<<endl;
  
  // Fill Pedestal Histograms
  int mydepth=0;


  for (int eta=0;eta<83;++eta)
    {
      for (int phi=0;phi<72;++phi)
	{
	  for (int depth=0;depth<4;++depth)
	    {
	      if (fabs(eta-41)>28 && depth>1) // shift HF cells back to their appropriate depths
		mydepth=depth-2;
	      else mydepth=depth;

	      if (pedcounts[eta][phi][depth]==0) continue;
	      
	      // When setting Bin Content, bins start at count of 1, not 0.
	      // Also, first bins around eta,phi are empty.
	      // Thus, eta,phi must be shifted by +2 (+1 for bin count, +1 to ignore empty row)
	      
	      if (fabs(eta-41)==29 && depth==2)
		// This value of eta is shared by HB, HE -- add their values together.  Maybe average them at some point instead?
		{
		  // raw pedestals
		  double myval= rawpedsum[eta][phi][depth]/pedcounts[eta][phi][2]; // HF
		  double myval2 = rawpedsum[eta][phi][0]/pedcounts[eta][phi][0]; // HE
		  RAW_PEDESTAL_MEAN[mydepth]->setBinContent(eta+2,phi+2,myval+myval2);
		  double RMS = 1.0*rawpedsum2[eta][phi][2]/pedcounts[eta][phi][2]-myval*myval;
		  
		  RMS=pow(fabs(RMS),0.5); // HF
		  double RMS2 = 1.0*rawpedsum2[eta][phi][0]/pedcounts[eta][phi][0]-myval2*myval2;
		  
		  RMS2=pow(fabs(RMS2),0.5); // HE
		  RAW_PEDESTAL_RMS[mydepth]->setBinContent(eta+2,phi+2,RMS+RMS2);
		  
		  // subtracted pedestals
		  myval= subpedsum[eta][phi][2]/pedcounts[eta][phi][2]; // HF
		  myval2 = subpedsum[eta][phi][0]/pedcounts[eta][phi][0]; // HE
		  SUB_PEDESTAL_MEAN[mydepth]->setBinContent(eta+2,phi+2,myval+myval2);
		  RMS = 1.0*subpedsum2[eta][phi][2]/pedcounts[eta][phi][2]-myval*myval;
		  
		  RMS=pow(fabs(RMS),0.5); // HF
		  RMS2 = 1.0*subpedsum2[eta][phi][0]/pedcounts[eta][phi][0]-myval2*myval2;
		  
		  RMS2=pow(fabs(RMS2),0.5); // HF
		  SUB_PEDESTAL_RMS[mydepth]->setBinContent(eta+2,phi+2,RMS+RMS2);
		}

	      else
		{
		  double myval= rawpedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
		  RAW_PEDESTAL_MEAN[mydepth]->setBinContent(eta+2,phi+2,myval);
		  double RMS = 1.0*rawpedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-myval*myval;
		  
		  RMS=pow(fabs(RMS),0.5); // use fabs just in case we run into rounding issues near 0
		  RAW_PEDESTAL_RMS[mydepth]->setBinContent(eta+2,phi+2,RMS);
		  
		  myval= subpedsum[eta][phi][depth]/pedcounts[eta][phi][depth];
		  SUB_PEDESTAL_MEAN[mydepth]->setBinContent(eta+2,phi+2,myval);
		  RMS = 1.0*subpedsum2[eta][phi][depth]/pedcounts[eta][phi][depth]-myval*myval;
		  
		  RMS=pow(fabs(RMS),0.5);
		  SUB_PEDESTAL_RMS[mydepth]->setBinContent(eta+2,phi+2,RMS);
		}

	      // Now add back in HE value at |eta|=29 (it gets overwritten by HF setBinContent
	      
	    } // for (int depth)
	} // for (int phi)
    } // for (int eta)

  double temp;
  int eta,phi;

  for (int ieta=1;ieta<=etaBins_;++ieta)
    {
      for (int iphi=1; iphi<=phiBins_;++iphi)
	{
	  eta=ieta+int(etaMin_)-1;
	  phi=iphi+int(phiMin_)-1;

	  // ignore unphysical values
	  if (phi==0) continue; 
	  if (phi>72) continue;
	  if (eta==0) continue; 
	  if (abs(eta)>41) continue;

	  // HB first
	  if (abs(eta)<17)
	    {

	      if (hbHists.check)
		temp=hbHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->GetBinContent(ieta,iphi);
	      if (hbHists.check && temp==0) // no digis found
		{
		  hbHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hbHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		}

	      if (abs(eta)>14) // last two rows of HB have two depths
		{
		  if (hbHists.check) temp=hbHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->GetBinContent(ieta,iphi);
		  if (hbHists.check && temp==0)
		    {
		      hbHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      hbHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		    }
		}
	    } // if (abs(eta)<17)  // HB Block

	  // HO loop -- depth = 4
	  if (abs(eta)<16)
	    {
	      if (hoHists.check) temp=hoHists.PROBLEMDIGICELLS_TEMP_DEPTH[3]->GetBinContent(ieta,iphi);
	      if (hoHists.check && temp==0)
		{
		  hoHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hoHists.PROBLEMDIGICELLS_DEPTH[3]->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS_DEPTH[3]->Fill(eta,phi,checkNevents_);
		}
	    } // if (abs(eta)<16)
	  
	  // HE loop (careful; phi values are odd only for eta>20)

	  if (abs(eta)==16) // at eta=16, HE depth=3
	    {
	      if (heHists.check) temp=heHists.PROBLEMDIGICELLS_TEMP_DEPTH[2]->GetBinContent(ieta,iphi);
	      if (heHists.check && temp==0)
		{
		  heHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  heHists.PROBLEMDIGICELLS_DEPTH[2]->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS_DEPTH[2]->Fill(eta,phi,checkNevents_);
		}
	    }

	  if (abs(eta)>16 && abs(eta)<30) // HE has depth = 1-2 for eta=18-29; 27-28 also depth=3
	    // This differs from documentation, which claimed that depth=3 for eta=28-29
	    {
	      if (abs(eta)<21 ||(abs(eta)>20 && (phi%2)==1)) // decreased phi segementation above eta=20
		{
		  if ( heHists.check) temp=heHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->GetBinContent(ieta,iphi);
		  if (heHists.check && temp==0)
		    {
		      heHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      heHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		    }
		}  
	      if (abs(eta)>17) // only one layer for HE in eta=17 -- skip it when filling depth=2
		{
		  if (abs(eta)>20 && (phi%2)==0)
		    continue;
		  if (heHists.check) temp=heHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->GetBinContent(ieta,iphi);
	
		  if (heHists.check && temp==0)
		    {
		      heHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      heHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		    }
		}

	      if (abs(eta)>26 && abs(eta)<29 && (phi%2)==1) // depth 3
		{
		  if (heHists.check) temp=heHists.PROBLEMDIGICELLS_TEMP_DEPTH[2]->GetBinContent(ieta,iphi);
		  
		  if (heHists.check && temp==0)
		    {
		      heHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      heHists.PROBLEMDIGICELLS_DEPTH[2]->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		      hcalHists.PROBLEMDIGICELLS_DEPTH[2]->Fill(eta,phi,checkNevents_);
		    }
		} // if (abs(eta)>26 && abs(eta)<29 ...)
	    } //  if (abs(eta)>15 && abs(eta)<30) // ends HE loop
	  


	  // HF Loop
	  if (abs(eta)>28 && abs(eta)<40 && (phi%2)==1)
	    {
	      // depth 1
	      if (hfHists.check) temp=hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->GetBinContent(ieta,iphi);
	
	      if (hfHists.check && temp==0)
		{
		  hfHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hfHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		}
	      //depth2
	      if (hfHists.check) temp=hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->GetBinContent(ieta,iphi);
	
	      if (hfHists.check && temp==0)
		{
		  hfHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hfHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		}
	    }

	  else if (abs(eta)>39 && (phi%4)==3)
	    {
	      // depth 1
	      if (hfHists.check) temp=hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[0]->GetBinContent(ieta,iphi);
	      
	      if (hfHists.check && temp==0)
		{

		  hfHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hfHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
		  hcalHists.PROBLEMDIGICELLS_DEPTH[0]->Fill(eta,phi,checkNevents_);
		}
	      
	      //depth2
	      if (hfHists.check) temp=hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[1]->GetBinContent(ieta,iphi);
	      
	      if (hfHists.check && temp==0)
		{
	      hfHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
	      hfHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
	      hcalHists.PROBLEMDIGICELLS->Fill(eta,phi,checkNevents_);
	      hcalHists.PROBLEMDIGICELLS_DEPTH[1]->Fill(eta,phi,checkNevents_);
		}
	    } // end HF loop

	} //for (int iphi=1; ...)
    } // for (int ieta=1; ...)
  
  // Reset temporary histograms
  hcalHists.PROBLEMDIGICELLS_TEMP->Reset();
  if (hbHists.check) hbHists.PROBLEMDIGICELLS_TEMP->Reset();
  if (heHists.check) heHists.PROBLEMDIGICELLS_TEMP->Reset();
  if (hoHists.check) hoHists.PROBLEMDIGICELLS_TEMP->Reset();
  if (hfHists.check) hfHists.PROBLEMDIGICELLS_TEMP->Reset();
  
  for (int d=0;d<4;++d)
    {
      hcalHists.PROBLEMDIGICELLS_TEMP_DEPTH[d]->Reset();
      if (hbHists.check) hbHists.PROBLEMDIGICELLS_TEMP_DEPTH[d]->Reset();
      if (heHists.check) heHists.PROBLEMDIGICELLS_TEMP_DEPTH[d]->Reset();
      if (hoHists.check) hoHists.PROBLEMDIGICELLS_TEMP_DEPTH[d]->Reset();
      if (hfHists.check) hfHists.PROBLEMDIGICELLS_TEMP_DEPTH[d]->Reset();
    }

} //void HcalDigiMonitor::CheckNevents(void)
