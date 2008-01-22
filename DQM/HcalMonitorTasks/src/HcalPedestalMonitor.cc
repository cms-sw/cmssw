#include "DQM/HcalMonitorTasks/interface/HcalPedestalMonitor.h"


HcalPedestalMonitor::HcalPedestalMonitor() { doPerChannel_ = false;   shape_=NULL; }

HcalPedestalMonitor::~HcalPedestalMonitor() {
}

void HcalPedestalMonitor::reset(){}

void HcalPedestalMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"PedestalMonitor";

  doPerChannel_ = ps.getUntrackedParameter<bool>("PedestalsPerChannel", false);
  doFCpeds_ = ps.getUntrackedParameter<bool>("PedestalsInFC", true);

  etaMax_ = ps.getUntrackedParameter<double>("MaxEta", 29.5);
  etaMin_ = ps.getUntrackedParameter<double>("MinEta", -29.5);
  etaBins_ = (int)(etaMax_ - etaMin_);

  phiMax_ = ps.getUntrackedParameter<double>("MaxPhi", 72.5);
  phiMin_ = ps.getUntrackedParameter<double>("MinPhi", -0.5);
  phiBins_ = (int)(phiMax_ - phiMin_);

  ievt_=0;

  if ( m_dbe ) {
    m_dbe->setCurrentFolder(baseFolder_);
    meEVT_ = m_dbe->bookInt("Pedestal Task Event Number");
    meEVT_->Fill(ievt_);

    MEAN_MAP_L1= m_dbe->book2D("Ped Mean Depth 1","Ped Mean Depth 1",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_L1= m_dbe->book2D("Ped RMS Depth 1","Ped RMS Depth 1",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_L2= m_dbe->book2D("Ped Mean Depth 2","Ped Mean Depth 2",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_L2= m_dbe->book2D("Ped RMS Depth 2","Ped RMS Depth 2",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_L3= m_dbe->book2D("Ped Mean Depth 3","Ped Mean Depth 3",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_L3= m_dbe->book2D("Ped RMS Depth 3","Ped RMS Depth 3",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    
    MEAN_MAP_L4= m_dbe->book2D("Ped Mean Depth 4","Ped Mean Depth 4",etaBins_,etaMin_,etaMax_,
			       phiBins_,phiMin_,phiMax_);
    RMS_MAP_L4= m_dbe->book2D("Ped RMS Depth 4","Ped RMS Depth 4",etaBins_,etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);


    char* type = "Ped Mean by Crate-Slot";
    MEAN_MAP_CR =m_dbe->book2D(type,type,21,-0.5,20.5,21,-0.5,20.5);
    type = "Ped RMS by Crate-Slot";
    RMS_MAP_CR =m_dbe->book2D(type,type,21,-0.5,20.5,21,-0.5,20.5);

    type = "Ped Mean by Fiber-Chan";
    MEAN_MAP_FIB =m_dbe->book2D(type,type,3,-0.5,2.5,9,-0.5,8.5);
    type = "Ped RMS by Fiber-Chan";
    RMS_MAP_FIB =m_dbe->book2D(type,type,3,-0.5,2.5,9,-0.5,8.5);
    type = "Pedestal Mean Reference Values";
    PEDESTAL_REFS = m_dbe->book1D(type,type,100,0,9);
    type = "Pedestal RMS Reference Values";
    WIDTH_REFS = m_dbe->book1D(type,type,100,0,3);

    
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    hbHists.ALLPEDS =  m_dbe->book1D("HB All Pedestal Values","HB All Pedestal Values",50,0,50);
    hbHists.PEDRMS  =  m_dbe->book1D("HB Pedestal RMS Values","HB Pedestal RMS Values",100,0,3);
    hbHists.PEDMEAN =  m_dbe->book1D("HB Pedestal Mean Values","HB Pedestal Mean Values",100,0,9);
    hbHists.NSIGMA  =  m_dbe->book1D("HB Normalized RMS Values","HB Normalized RMS Values",100,0,5);
    hbHists.SUBMEAN =  m_dbe->book1D("HB Subtracted Mean Values","HB Subtracted Mean Values",100,-2.5,2.5);
    hbHists.CAPIDRMS  =  m_dbe->book1D("HB CapID RMS Variance","HB CapID RMS Variance",50,0,0.5);
    hbHists.CAPIDMEAN =  m_dbe->book1D("HB CapID Mean Variance","HB CapID Mean Variance",50,0,3);
    hbHists.QIERMS  =  m_dbe->book1D("HB QIE RMS Values","HB QIE RMS Values",50,0,3);
    hbHists.QIEMEAN =  m_dbe->book1D("HB QIE Mean Values","HB QIE Mean Values",50,0,10);
    hbHists.ERRGEO =  m_dbe->book2D("HB Pedestal Geo Error Map","HB Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hbHists.ERRELEC =  m_dbe->book2D("HB Pedestal Elec Error Map","HB Pedestal Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);
    type = "HB Pedestal Mean Reference Values";
    hbHists.PEDESTAL_REFS = m_dbe->book1D(type,type,100,0,9);
    type = "HB Pedestal RMS Reference Values";
    hbHists.WIDTH_REFS = m_dbe->book1D(type,type,50,0,3);



    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    heHists.ALLPEDS =  m_dbe->book1D("HE All Pedestal Values","HE All Pedestal Values",50,0,50);
    heHists.PEDRMS  =  m_dbe->book1D("HE Pedestal RMS Values","HE Pedestal RMS Values",100,0,3);
    heHists.PEDMEAN =  m_dbe->book1D("HE Pedestal Mean Values","HE Pedestal Mean Values",100,0,9);
    heHists.NSIGMA  =  m_dbe->book1D("HE Normalized RMS Values","HE Normalized RMS Values",100,0,5);
    heHists.SUBMEAN =  m_dbe->book1D("HE Subtracted Mean Values","HE Subtracted Mean Values",100,-2.5,2.5);
    heHists.CAPIDRMS  =  m_dbe->book1D("HE CapID RMS Variance","HE CapID RMS Variance",50,0,0.5);
    heHists.CAPIDMEAN =  m_dbe->book1D("HE CapID Mean Variance","HE CapID Mean Variance",50,0,3);
    heHists.QIERMS  =  m_dbe->book1D("HE QIE RMS Values","HE QIE RMS Values",50,0,3);
    heHists.QIEMEAN =  m_dbe->book1D("HE QIE Mean Values","HE QIE Mean Values",50,0,10);
    heHists.ERRGEO =  m_dbe->book2D("HE Pedestal Geo Error Map","HE Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    heHists.ERRELEC =  m_dbe->book2D("HE Pedestal Elec Error Map","HE Pedestal Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);
    type = "HE Pedestal Mean Reference Values";
    heHists.PEDESTAL_REFS = m_dbe->book1D(type,type,100,0,9);
    type = "HE Pedestal RMS Reference Values";
    heHists.WIDTH_REFS = m_dbe->book1D(type,type,50,0,3);

    
    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    hfHists.ALLPEDS =  m_dbe->book1D("HF All Pedestal Values","HF All Pedestal Values",50,0,50);
    hfHists.PEDRMS  =  m_dbe->book1D("HF Pedestal RMS Values","HF Pedestal RMS Values",100,0,3);
    hfHists.NSIGMA  =  m_dbe->book1D("HF Normalized RMS Values","HF Normalized RMS Values",100,0,5);
    hfHists.SUBMEAN =  m_dbe->book1D("HF Subtracted Mean Values","HF Subtracted Mean Values",100,-2.5,2.5);
    hfHists.PEDMEAN =  m_dbe->book1D("HF Pedestal Mean Values","HF Pedestal Mean Values",100,0,9);
    hfHists.CAPIDRMS  =  m_dbe->book1D("HF CapID RMS Variance","HF CapID RMS Variance",50,0,0.5);
    hfHists.CAPIDMEAN =  m_dbe->book1D("HF CapID Mean Variance","HF CapID Mean Variance",50,0,3);
    hfHists.QIERMS  =  m_dbe->book1D("HF QIE RMS Values","HF QIE RMS Values",50,0,3);
    hfHists.QIEMEAN =  m_dbe->book1D("HF QIE Mean Values","HF QIE Mean Values",50,0,10);
    hfHists.ERRGEO =  m_dbe->book2D("HF Pedestal Geo Error Map","HF Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hfHists.ERRELEC =  m_dbe->book2D("HF Pedestal Elec Error Map","HF Pedestal Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);
    type = "HF Pedestal Mean Reference Values";
    hfHists.PEDESTAL_REFS = m_dbe->book1D(type,type,100,0,9);
    type = "HF Pedestal RMS Reference Values";
    hfHists.WIDTH_REFS = m_dbe->book1D(type,type,50,0,3);

    

    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    hoHists.ALLPEDS =  m_dbe->book1D("HO All Pedestal Values","HO All Pedestal Values",50,0,50);
    hoHists.PEDRMS  =  m_dbe->book1D("HO Pedestal RMS Values","HO Pedestal RMS Values",100,0,3);
    hoHists.PEDMEAN =  m_dbe->book1D("HO Pedestal Mean Values","HO Pedestal Mean Values",100,0,9);
    hoHists.NSIGMA  =  m_dbe->book1D("HO Normalized RMS Values","HO Normalized RMS Values",100,0,5);
    hoHists.SUBMEAN =  m_dbe->book1D("HO Subtracted Mean Values","HO Subtracted Mean Values",100,-2.5,2.5);
    hoHists.CAPIDRMS  =  m_dbe->book1D("HO CapID RMS Variance","HO CapID RMS Variance",50,0,0.5);
    hoHists.CAPIDMEAN =  m_dbe->book1D("HO CapID Mean Variance","HO CapID Mean Variance",50,0,3);
    hoHists.QIERMS  =  m_dbe->book1D("HO QIE RMS Values","HO QIE RMS Values",50,0,3);
    hoHists.QIEMEAN =  m_dbe->book1D("HO QIE Mean Values","HO QIE Mean Values",50,0,10);
    hoHists.ERRGEO =  m_dbe->book2D("HO Pedestal Geo Error Map","HO Pedestal Geo Error Map",etaBins_,etaMin_,etaMax_,phiBins_,phiMin_,phiMax_);
    hoHists.ERRELEC =  m_dbe->book2D("HO Pedestal Elec Error Map","HO Pedestal Elec Error Map",21,-0.5,20.5,21,-0.5,20.5);
    type = "HO Pedestal Mean Reference Values";
    hoHists.PEDESTAL_REFS = m_dbe->book1D(type,type,100,0,9);
    type = "HO Pedestal RMS Reference Values";
    hoHists.WIDTH_REFS = m_dbe->book1D(type,type,50,0,3);

}
  
  outputFile_ = ps.getUntrackedParameter<string>("PedestalFile", "");
  if ( outputFile_.size() != 0 ) {
    if(fVerbosity) cout << "Hcal Pedestal Calibrations will be saved to " << outputFile_.c_str() << endl;
  }

  return;
}

void HcalPedestalMonitor::processEvent(const HBHEDigiCollection& hbhe,
				       const HODigiCollection& ho,
				       const HFDigiCollection& hf,
				       const HcalDbService& cond){
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  if(!shape_) shape_ = cond.getHcalShape(); // this one is generic

  if(!m_dbe) { 
    if(fVerbosity) printf("HcalPedestalMonitor::processEvent   DaqMonitorBEInterface not instantiated!!!\n");  
    return; 
  }

  CaloSamples tool;  
  try{    
    for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); j++){
      
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      const HcalPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
      cond.makeHcalCalibration(digi.id(), &calibs_);
      detID_.clear(); capID_.clear(); pedVals_.clear();
      if(doFCpeds_){
	channelCoder_ = cond.getHcalCoder(digi.id());
	HcalCoderDb coderDB(*channelCoder_, *shape_);
	coderDB.adc2fC(digi,tool);
      }

      if((HcalSubdetector)(digi.id().subdet())==HcalBarrel){
	for(int capID=0; capID<4; capID++){
	  float width=0;
	  if(pedw) width = pedw->getWidth(capID);
	  hbHists.PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  hbHists.WIDTH_REFS->Fill(width);
	  PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  WIDTH_REFS->Fill(width);
	}
	for (int i=0; i<digi.size(); i++) {
	  if(doFCpeds_) pedVals_.push_back(tool[i]);
	  else pedVals_.push_back(digi.sample(i).adc());
	  hbHists.ALLPEDS->Fill(pedVals_[i]);
	  detID_.push_back(digi.id());
	  capID_.push_back(digi.sample(i).capid());
	}
	if(doPerChannel_) perChanHists(0,detID_,capID_,pedVals_,
				       hbHists.PEDVALS,hbHists.SUBVALS, baseFolder_);
	
      }   
      else if((HcalSubdetector)(digi.id().subdet())==HcalEndcap){
	for(int capID=0; capID<4; capID++){
	  float width=0;
	  if(pedw) width = pedw->getWidth(capID);
	  heHists.PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  heHists.WIDTH_REFS->Fill(width);
	  PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  WIDTH_REFS->Fill(width);
	}

	for (int i=0; i<digi.size(); i++) {
	  if(doFCpeds_) pedVals_.push_back(tool[i]);
	  else pedVals_.push_back(digi.sample(i).adc());
	  detID_.push_back(digi.id());
	  capID_.push_back(digi.sample(i).capid());
	  heHists.ALLPEDS->Fill(pedVals_[i]);
	}
	if(doPerChannel_) perChanHists(1,detID_,capID_,pedVals_,
				       heHists.PEDVALS,heHists.SUBVALS, baseFolder_);
      }
    }
  } catch (...) {
    printf("HcalPedestalMonitor::processEvent  No HBHE Digis.\n");
  }
  
  try{
    for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); j++){
      const HODataFrame digi = (const HODataFrame)(*j);	
      const HcalPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
      cond.makeHcalCalibration(digi.id(), &calibs_);
      detID_.clear(); capID_.clear(); pedVals_.clear();
      if(doFCpeds_){
	channelCoder_ = cond.getHcalCoder(digi.id());
	HcalCoderDb coderDB(*channelCoder_, *shape_);
	coderDB.adc2fC(digi,tool);
      }
      for(int capID=0; capID<4; capID++){
	  float width=0;
	  if(pedw) width = pedw->getWidth(capID);
	  hoHists.PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  hoHists.WIDTH_REFS->Fill(width);
	  PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  WIDTH_REFS->Fill(width);
	}

	for (int i=0; i<digi.size(); i++) {
	  if(doFCpeds_) pedVals_.push_back(tool[i]);
	  else pedVals_.push_back(digi.sample(i).adc());
	  detID_.push_back(digi.id());
	  capID_.push_back(digi.sample(i).capid());
	  hoHists.ALLPEDS->Fill(pedVals_[i]);
	}
	if(doPerChannel_) perChanHists(2,detID_,capID_,pedVals_,
				       hoHists.PEDVALS,hoHists.SUBVALS, baseFolder_);
	
    }
  } catch (...) {
    if(fVerbosity) cout << "HcalPedestalMonitor::processEvent  No HO Digis." << endl;
  }
  
  try{
    for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); j++){
      const HFDataFrame digi = (const HFDataFrame)(*j);	
      const HcalPedestalWidth* pedw = cond.getPedestalWidth(digi.id());
      cond.makeHcalCalibration(digi.id(), &calibs_);
      detID_.clear(); capID_.clear(); pedVals_.clear();
      if(doFCpeds_){
	channelCoder_ = cond.getHcalCoder(digi.id());
	HcalCoderDb coderDB(*channelCoder_, *shape_);
	coderDB.adc2fC(digi,tool);
      }
      for(int capID=0; capID<4; capID++){
	  float width=0;
	  if(pedw) width = pedw->getWidth(capID);
	  hfHists.PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  hfHists.WIDTH_REFS->Fill(width);
	  PEDESTAL_REFS->Fill(calibs_.pedestal(capID));
	  WIDTH_REFS->Fill(width);
      }
      
      for (int i=0; i<digi.size(); i++) {
	if(doFCpeds_) pedVals_.push_back(tool[i]);
	else pedVals_.push_back(digi.sample(i).adc());
	detID_.push_back(digi.id());
	capID_.push_back(digi.sample(i).capid());
	hfHists.ALLPEDS->Fill(pedVals_[i]);
      }
      if(doPerChannel_) perChanHists(3,detID_,capID_,pedVals_,
				     hfHists.PEDVALS,hfHists.SUBVALS, baseFolder_);
      
    }
  } catch (...) {
    if(fVerbosity) cout << "HcalPedestalMonitor::processEvent  No HF Digis." << endl;
  }
  

  return;
}

void HcalPedestalMonitor::done(){

  return;
}

void HcalPedestalMonitor::perChanHists(int id, 
				       vector<HcalDetId> detID, 
				       vector<int> capID, 
				       vector<float> peds,
				       map<HcalDetId, map<int, MonitorElement*> > &toolP, 
				       map<HcalDetId, map<int, MonitorElement*> > &toolS, string baseFolder) {
  

  string type = "HB";
  if(id==1) type = "HE"; 
  else if(id==2) type = "HF"; 
  else if(id==3) type = "HO"; 
  
  if(m_dbe) m_dbe->setCurrentFolder(baseFolder+"/"+type);


  for(unsigned int d=0; d<detID.size(); d++){
    HcalDetId detid = detID[d];
    int capid = capID[d];
    float pedVal = peds[d];
    //outer iteration
    bool gotit=false;
    if(REG[detid]) gotit=true;
    
    if(gotit){
      //inner iteration
      map<int, MonitorElement*> _mei = toolP[detid];
      if(_mei[capid]==NULL){
	if(fVerbosity) printf("HcalPedestalAnalysis::perChanHists  This histo is NULL!!??\n");
      }
      else _mei[capid]->Fill(pedVal);
      
      _mei = toolS[detid];
      if(_mei[capid]==NULL){
	if(fVerbosity) printf("HcalPedestalAnalysis::perChanHists  This histo is NULL!!??\n");
      }
      else _mei[capid]->Fill(pedVal-calibs_.pedestal(capid));
    }
    else{
      if(m_dbe){
	map<int,MonitorElement*> insertP;
	map<int,MonitorElement*> insertS;
	
	for(int i=0; i<4; i++){
	  char name[1024];
	  sprintf(name,"%s Pedestal Value (ADC) ieta=%d iphi=%d depth=%d CAPID=%d",
		  type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i);      
	  insertP[i] =  m_dbe->book1D(name,name,10,-0.5,9.5);
	  
	  sprintf(name,"%s Pedestal Value (Subtracted) ieta=%d iphi=%d depth=%d CAPID=%d",
		  type.c_str(),detid.ieta(),detid.iphi(),detid.depth(),i);      
	  insertS[i] =  m_dbe->book1D(name,name,10,-5,5);	
	}
	insertP[capid]->Fill(pedVal);
	insertS[capid]->Fill(pedVal-calibs_.pedestal(capid));
	toolP[detid] = insertP;
	toolS[detid] = insertS;
      }
      REG[detid] = true;
    }
  }
}
