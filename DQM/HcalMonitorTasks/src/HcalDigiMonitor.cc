#include "DQM/HcalMonitorTasks/interface/HcalDigiMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include <cmath>

#include "FWCore/Common/interface/TriggerNames.h" 
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

// constructor
HcalDigiMonitor::HcalDigiMonitor(const edm::ParameterSet& ps):HcalBaseDQMonitor(ps)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","DigiMonitor_Hcal"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);
  digiLabel_             = ps.getUntrackedParameter<edm::InputTag>("digiLabel");
  FEDRawDataCollection_  = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection");
  shapeThresh_           = ps.getUntrackedParameter<int>("shapeThresh",20);
  //shapeThresh_ is used for plotting pulse shapes for all digis with pedestal-subtracted ADC sum > shapeThresh_;
  shapeThreshHB_ = ps.getUntrackedParameter<int>("shapeThreshHB",shapeThresh_);
  shapeThreshHE_ = ps.getUntrackedParameter<int>("shapeThreshHE",shapeThresh_);
  shapeThreshHF_ = ps.getUntrackedParameter<int>("shapeThreshHF",shapeThresh_);
  shapeThreshHO_ = ps.getUntrackedParameter<int>("shapeThreshHO",shapeThresh_);
  
  hltresultsLabel_       = ps.getUntrackedParameter<edm::InputTag>("HLTResultsLabel");
  MinBiasHLTBits_        = ps.getUntrackedParameter<std::vector<std::string> >("MinBiasHLTBits");
  excludeHORing2_       = ps.getUntrackedParameter<bool>("excludeHORing2",false);
  excludeHO1P02_        = ps.getUntrackedParameter<bool>("excludeHO1P02",false);
  excludeBadQPLLs_      = ps.getUntrackedParameter<bool>("excludeBadQPLL",false);

  if (debug_>0)
    std::cout <<"<HcalDigiMonitor> Digi shape ADC threshold set to: >" << shapeThresh_ <<" counts above nominal pedestal (3*10)"<< std::endl;
  
  // Specify which tests to run when looking for problem digis
  digi_checkoccupancy_ = ps.getUntrackedParameter<bool>("checkForMissingDigis",false); // off by default -- checked by dead cell monitor
  digi_checkcapid_     = ps.getUntrackedParameter<bool>("checkCapID",true);
  digi_checkdigisize_  = ps.getUntrackedParameter<bool>("checkDigiSize",true);
  digi_checkadcsum_    = ps.getUntrackedParameter<bool>("checkADCsum",true);
  digi_checkdverr_     = ps.getUntrackedParameter<bool>("checkDVerr",true);
  mindigisizeHBHE_     = ps.getUntrackedParameter<int>("minDigiSizeHBHE",1);
  maxdigisizeHBHE_     = ps.getUntrackedParameter<int>("maxDigiSizeHBHE",10);
  mindigisizeHO_       = ps.getUntrackedParameter<int>("minDigiSizeHO",1);
  maxdigisizeHO_       = ps.getUntrackedParameter<int>("maxDigiSizeHO",10);
  mindigisizeHF_       = ps.getUntrackedParameter<int>("minDigiSizeHF",1);
  maxdigisizeHF_       = ps.getUntrackedParameter<int>("maxDigiSizeHF",10);


  badChannelStatusMask_   = ps.getUntrackedParameter<int>("BadChannelStatusMask",
                                                          ps.getUntrackedParameter<int>("BadChannelStatusMask",
											(1<<HcalChannelStatus::HcalCellDead)));  // identify channel status values to mask
  if (debug_>1)
    {
      std::cout <<"<HcalDigiMonitor> Checking for the following problems:"<<std::endl; 
      if (digi_checkcapid_) std::cout <<"\tChecking that cap ID rotation is correct;"<<std::endl;
      if (digi_checkdigisize_)
	{
	  std::cout <<"\tChecking that HBHE digi size is between ["<<mindigisizeHBHE_<<" - "<<maxdigisizeHBHE_<<"];"<<std::endl;
	  std::cout <<"\tChecking that HO digi size is between ["<<mindigisizeHO_<<" - "<<maxdigisizeHO_<<"];"<<std::endl;
	  std::cout <<"\tChecking that HF digi size is between ["<<mindigisizeHF_<<" - "<<maxdigisizeHF_<<"];"<<std::endl;
	}
      if (digi_checkadcsum_) std::cout <<"\tChecking that ADC sum of digi is greater than 0;"<<std::endl; 
      if (digi_checkdverr_) std::cout <<"\tChecking that data valid bit is true and digi error bit is false;\n"<<std::endl;
    }
  
  shutOffOrbitTest_ = ps.getUntrackedParameter<bool>("shutOffOrbitTest",false);
  DigiMonitor_ExpectedOrbitMessageTime_=ps.getUntrackedParameter<int>("ExpectedOrbitMessageTime",3559); // -1 means that orbit mismatches won't be checked

  HFtiming_totaltime2D=0;
  HFtiming_occupancy2D=0;
  HFtiming_etaProfile=0;
  HFP_shape=0;
  HFM_shape=0;
  setupDone_=false;

  // register for data access
  tok_raw_ = consumes<FEDRawDataCollection>(FEDRawDataCollection_);
  tok_hbhe_ = consumes<HBHEDigiCollection>(digiLabel_);
  tok_ho_ = consumes<HODigiCollection>(digiLabel_);
  tok_hf_ = consumes<HFDigiCollection>(digiLabel_);
  tok_unpack_ = consumes<HcalUnpackerReport>(digiLabel_);
  tok_trigger_ = consumes<edm::TriggerResults>(hltresultsLabel_);
  tok_hfrec_   = consumes<HFRecHitCollection>(ps.getUntrackedParameter<edm::InputTag>("hfRechitLabel"));

  //set Token(-s)
  dcsStatusToken_ = consumes<DcsStatusCollection>(std::string("scalersRawToDigi"));
  FEDRawDataCollectionToken_ = consumes<FEDRawDataCollection>(FEDRawDataCollection_);
}

// destructor
HcalDigiMonitor::~HcalDigiMonitor() {}

// Checks capid rotation; returns false if no problems with rotation
static bool bitUpset(int last, int now){
  if(last ==-1) return false;
  int v = last+1; 
  if(v==4) v=0;
  if(v==now) return false;
  return true;
} // static bool bitUpset(...)

/*void HcalDigiMonitor::cleanup()
{
  // Need to add code to clear out subfolders as well?
  if (debug_>0) std::cout <<"HcalDigiMonitor::cleanup()"<<std::endl;
  if (!enableCleanup_) return;
  if (dbe_)
    {
      // removeContents doesn't remove subdirectories
      dbe_->setCurrentFolder(subdir_);
      dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"digi_parameters");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/bad_digi_occupancy");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/1D_digi_plots");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/badcapID");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/data_invalid_error");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/bad_reportUnpackerErrors");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/baddigisize");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"digi_info");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/badfibBCNoff");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"good_digis/1D_digi_plots");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"good_digis/digi_occupancy");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis/bad_digi_occupancy");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"bad_digis");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"good_digis/");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"digi_info/HB");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"digi_info/HE");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"digi_info/HO");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"digi_info/HF");  dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"LSvalues");
      dbe_->removeContents();
    } // if(dbe_)

}*/ // void HcalDigiMonitor::cleanup();


void HcalDigiMonitor::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  // Anything to do here?
}

void HcalDigiMonitor::endJob()
{
  if (debug_>0) std::cout <<"HcalDigiMonitor::endJob()"<<std::endl;
  if (enableCleanup_) cleanup(); // when do we force cleanup?
}


void HcalDigiMonitor::setup(DQMStore::IBooker &ib)
{
  if (setupDone_)
    return;
  setupDone_=true;
  // Call base class setup
  HcalBaseDQMonitor::setup(ib);

  /******* Set up all histograms  ********/
  if (debug_>1)
    std::cout <<"<HcalDigiMonitor::setup>  Setting up histograms"<<std::endl;

  std::ostringstream name;
  ib.setCurrentFolder(subdir_);
  
  ib.setCurrentFolder(subdir_+"digi_parameters");
  MonitorElement* ExpectedOrbit = ib.bookInt("ExpectedOrbitMessageTime");
  ExpectedOrbit->Fill(DigiMonitor_ExpectedOrbitMessageTime_);

  MonitorElement* shapeT = ib.bookInt("DigiShapeThresh");
  shapeT->Fill(shapeThresh_);
  MonitorElement* shapeTHB = ib.bookInt("DigiShapeThreshHB");
  shapeTHB->Fill(shapeThreshHB_);
  MonitorElement* shapeTHE = ib.bookInt("DigiShapeThreshHE");
  shapeTHE->Fill(shapeThreshHE_);
  MonitorElement* shapeTHO = ib.bookInt("DigiShapeThreshHO");
  shapeTHO->Fill(shapeThreshHO_);
  MonitorElement* shapeTHF = ib.bookInt("DigiShapeThreshHF");
  shapeTHF->Fill(shapeThreshHF_);
  
  ib.setCurrentFolder(subdir_+"bad_digis/bad_digi_occupancy");
  SetupEtaPhiHists(ib,DigiErrorsByDepth,"Bad Digi Map","");
  ib.setCurrentFolder(subdir_+"bad_digis/1D_digi_plots");
  ProblemsVsLB=ib.bookProfile("BadDigisVsLB","Number Bad Digis vs Luminosity block;Lumi block;# of Bad digis",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);
  ProblemsVsLB_HB=ib.bookProfile("HB Bad Quality Digis vs LB","HB Bad Quality Digis vs Luminosity Block",
				     NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				     100,0,10000);   
  ProblemsVsLB_HE=ib.bookProfile("HE Bad Quality Digis vs LB","HE Bad Quality Digis vs Luminosity Block",
				     NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				     100,0,10000);
  ProblemsVsLB_HO=ib.bookProfile("HO Bad Quality Digis vs LB","HO Bad Quality Digis vs Luminosity Block",
				     NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				     100,0,10000);
  ProblemsVsLB_HF=ib.bookProfile("HF Bad Quality Digis vs LB","HF Bad Quality Digis vs Luminosity Block",
				     NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				     100,0,10000);
  ProblemsVsLB_HBHEHF=ib.bookProfile("HBHEHF Bad Quality Digis vs LB","HBHEHF Bad Quality Digis vs Luminosity Block",
				     NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				     100,0,10000);

  ProblemDigisInLastNLB_HBHEHF_alarm=ib.book1D("ProblemDigisInLastNLB_HBHEHF_alarm",
					      "Total Number of ProblemDigis HBHEHF in last 10 LS. Last bin contains OverFlow",
					      100,0,100);


  if (makeDiagnostics_) 
    {
      // by default, unpacked digis won't have these errors
      ib.setCurrentFolder(subdir_+"diagnostics/bad_digis/badcapID");
      SetupEtaPhiHists(ib,DigiErrorsBadCapID," Digis with Bad Cap ID Rotation", "");
      ib.setCurrentFolder(subdir_+"diagnostics/bad_digis/data_invalid_error");
      SetupEtaPhiHists(ib,DigiErrorsDVErr," Digis with Data Invalid or Error Bit Set", "");
    }
  
  if (Online_)
    {
      // Special histograms for Pawel's timing study
      ib.setCurrentFolder(subdir_+"HFTimingStudy");
      HFtiming_etaProfile=ib.bookProfile("HFTiming_etaProfile","HFTiming Eta Profile;ieta;average time (time slice)",83,-41.5,41.5,200,0,10);
      HFP_shape=ib.book1D("HFP_signal_shape","HFP signal shape",10,-0.5,9.5);
      HFM_shape=ib.book1D("HFM_signal_shape","HFM signal shape",10,-0.5,9.5);
      ib.setCurrentFolder(subdir_+"HFTimingStudy/sumplots");
      HFtiming_totaltime2D=ib.book2D("HFTiming_Total_Time","HFTiming Total Time",83,-41.5,41.5,72,0.5,72.5);
      HFtiming_occupancy2D=ib.book2D("HFTiming_Occupancy","HFTiming Occupancy",83,-41.5,41.5,72,0.5,72.5);
    }
  
  ib.setCurrentFolder(subdir_+"bad_digis/bad_reportUnpackerErrors");
  SetupEtaPhiHists(ib,DigiErrorsUnpacker," Bad Unpacker Digis", "");
  
  ib.setCurrentFolder(subdir_+"bad_digis/baddigisize");
  SetupEtaPhiHists(ib,DigiErrorsBadDigiSize," Digis with Bad Size", "");
  
  ib.setCurrentFolder(subdir_+"digi_info");
  
  h_valid_digis=ib.book1D("ValidEvents","Events with minimum number of valid digis",2,-0.5,1.5);
  h_valid_digis->setBinLabel(1,"Valid");
  h_valid_digis->setBinLabel(2,"Invalid");
  
  h_invalid_orbitnumMod103=ib.book1D("InvalidDigiEvents_ORN","Orbit Number (mod 103) for Events with Many Unpacker Errors",103,-0.5,102.5);
  h_invalid_bcn=ib.book1D("InvalidDigiEvents_BCN","Bunch Crossing Number fo Events with Many Unpacker Errors",3464,-0.5,3563.5);

  DigiSize = ib.book2D("Digi Size", "Digi Size",4,0,4,20,-0.5,19.5);
  DigiSize->setBinLabel(1,"HB",1);
  DigiSize->setBinLabel(2,"HE",1);
  DigiSize->setBinLabel(3,"HO",1);
  DigiSize->setBinLabel(4,"HF",1);
  DigiSize->setAxisTitle("Subdetector",1);
  DigiSize->setAxisTitle("Digi Size",2);
  
  DigiExpectedSize = ib.book2D("Digi Expected Size", "Digi Expected Size",3,0,3,20,-0.5,19.5);
  DigiExpectedSize->setBinLabel(1,"HBHE",1);
  DigiExpectedSize->setBinLabel(2,"HO",1);
  DigiExpectedSize->setBinLabel(3,"HF",1);
  DigiExpectedSize->setAxisTitle("Subdetector",1);
  DigiExpectedSize->setAxisTitle("Digi Expected Size from HTR",2);
  
  ib.setCurrentFolder(subdir_+"bad_digis/badfibBCNoff");
  SetupEtaPhiHists(ib,DigiErrorsBadFibBCNOff," Digis with non-zero Fiber Orbit Msg Idle BCN Offsets", "");
  
  ib.setCurrentFolder(subdir_+"good_digis/1D_digi_plots");
  HBocc_vs_LB=ib.bookProfile("HBoccVsLB","HB digi occupancy vs Luminosity Block;Lumi block;# of Good digis",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 0,2600);
  HEocc_vs_LB=ib.bookProfile("HEoccVsLB","HE digi occupancy vs Luminosity Block;Lumi block;# of Good digis",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 0,2600);
  HOocc_vs_LB=ib.bookProfile("HOoccVsLB","HO digi occupancy vs Luminosity Block;Lumi block;# of Good digis",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 0,2200);
  HFocc_vs_LB=ib.bookProfile("HFoccVsLB","HF digi occupancy vs Luminosity Block;Lumi block;# of Good digis",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 0,1800);
  
  ib.setCurrentFolder(subdir_+"good_digis/digi_occupancy");
  SetupEtaPhiHists(ib,DigiOccupancyByDepth," Digi Eta-Phi Occupancy Map","");
  DigiOccupancyPhi= ib.book1D("Digi Phi Occupancy Map",
				  "Digi Phi Occupancy Map;i#phi;# of Events",
				  72,0.5,72.5);
  DigiOccupancyEta= ib.book1D("Digi Eta Occupancy Map",
				  "Digi Eta Occupancy Map;i#eta;# of Events",
				  83,-41.5,41.5);
  DigiOccupancyVME = ib.book2D("Digi VME Occupancy Map",
				   "Digi VME Occupancy Map;HTR Slot;VME Crate Id",
				   40,-0.25,19.75,36,-0.5,35.5);
  
  DigiOccupancySpigot = ib.book2D("Digi Spigot Occupancy Map",
				      "Digi Spigot Occupancy Map;Spigot;DCC Id",
				      HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				      36,-0.5,35.5);
  
  ib.setCurrentFolder(subdir_+"bad_digis/bad_digi_occupancy");
  DigiErrorVME = ib.book2D("Digi VME Error Map",
			       "Digi VME Error Map;HTR Slot;VME Crate Id",
			       40,-0.25,19.75,18,-0.5,17.5);
  
  DigiErrorSpigot = ib.book2D("Digi Spigot Error Map",
				  "Digi Spigot Error Map;Spigot;DCC Id",
				  HcalDCCHeader::SPIGOT_COUNT,-0.5,HcalDCCHeader::SPIGOT_COUNT-0.5,
				  36,-0.5,35.5);
  
  ib.setCurrentFolder(subdir_+"bad_digis");
  int nbins = sizeof(bins_cellcount_new)/sizeof(float)-1;

  DigiBQ = ib.book1D("NumBadQualDigis","Number Bad Qual Digis within Digi Collection",nbins, bins_cellcount_new);

  nbins=sizeof(bins_fraccount_new)/sizeof(float)-1;

  DigiBQFrac =  ib.book1D("Bad Digi Fraction","Bad Digi Fraction;Bad Quality Digi Fraction for digis in collection; # of Events",
			     nbins, bins_fraccount_new);
  
  nbins = sizeof(bins_cellcount_new)/sizeof(float)-1;
  DigiUnpackerErrorCount = ib.book1D("Unpacker Error Count", "Number of Bad Digis from Unpacker; Bad Unpacker Digis; # of Events",nbins, bins_cellcount_new);
  
  nbins=sizeof(bins_fraccount_new)/sizeof(float)-1;
  DigiUnpackerErrorFrac = ib.book1D("Unpacker Bad Digi Fraction", 
				       "Bad Digis From Unpacker/ (Bad Digis From Unpacker + Good Digis); Bad Unpacker Fraction; # of Events",
				       nbins,bins_fraccount_new);
  
  ib.setCurrentFolder(subdir_+"good_digis/");
  DigiNum = ib.book1D("NumGoodDigis","Number of Digis;# of Good Digis;# of Events",DIGI_NUM+1,-0.5,DIGI_NUM+1-0.5);
    
  setupSubdetHists(ib,hbHists,"HB");
  setupSubdetHists(ib,heHists,"HE");
  setupSubdetHists(ib,hoHists,"HO");
  setupSubdetHists(ib,hfHists,"HF");

  this->reset();
  return;
} // void HcalDigiMonitor::setup()

void HcalDigiMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  HcalBaseDQMonitor::bookHistograms(ib,run,c);
  if (mergeRuns_ && tevt_>0) return; // don't reset counters if merging runs

  if (debug_>1) std::cout <<"\t<HcalDigiMonitor::bookHistograms> Getting conditions from DB!"<<std::endl;
  c.get<HcalDbRecord>().get(conditions_);

  // Get all pedestals by Cap ID
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get("withTopo",p);
  const HcalChannelQuality *chanquality= p.product();
  std::vector<DetId> mydetids = chanquality->getAllChannels();
  PedestalsByCapId_.clear();

  for (std::vector<DetId>::const_iterator chan = mydetids.begin();chan!=mydetids.end();++chan)
    {
      if (chan->det()!=DetId::Hcal) continue; // not hcal
      std::vector <double> peds;  // could be ints, right?
      peds.clear();
      HcalCalibrations calibs=conditions_->getHcalCalibrations(*chan);
      const HcalQIECoder* channelCoder = conditions_->getHcalCoder(*chan);
      const HcalQIEShape* shape = conditions_->getHcalShape(channelCoder);
      //double total=0; // use this is we want to calculate average pedestal value
      for (int capid=0;capid<4;++capid)
	{
	  // temp_ADC should be an int, right?
	  double temp_ADC=channelCoder->adc(*shape,(float)calibs.pedestal(capid),capid);
	  peds.push_back(temp_ADC);
	  //total=total+temp_ADC;
	}
      //for (int capid=0;capid<4;++capid) peds.push_back(total/4.); // use this if we just want to use average value
      PedestalsByCapId_[*chan]=peds;
    } // loop on DetIds

  if (tevt_==0) this->setup(ib); // create all histograms; not necessary if merging runs together
  if (mergeRuns_==false) this->reset(); // call reset at start of all runs

  // Get known dead cells for this run
  KnownBadCells_.clear();
  if (badChannelStatusMask_>0)
    {
      edm::ESHandle<HcalChannelQuality> p;
      c.get<HcalChannelQualityRcd>().get("withTopo",p);
      const HcalChannelQuality* chanquality= p.product();
      std::vector<DetId> mydetids = chanquality->getAllChannels();
      for (std::vector<DetId>::const_iterator i = mydetids.begin();
	   i!=mydetids.end();
	   ++i)
	{
	  if (i->det()!=DetId::Hcal) continue; // not an hcal cell
	  HcalDetId id=HcalDetId(*i);
	  int status=(chanquality->getValues(id))->getValue();
	  if ((status & badChannelStatusMask_))
	    {
	      KnownBadCells_[id.rawId()]=status;
	    }
	} 
    } // if (badChannelStatusMask_>0)

} // void HcalDigiMonitor::bookHistograms()


void HcalDigiMonitor::setupSubdetHists(DQMStore::IBooker &ib, DigiHists& hist, std::string subdet)
{
  std::stringstream name;
  int nChan=0;
  if (subdet=="HB" || subdet=="HE") nChan=2592;
  else if (subdet == "HO") nChan=2160;
  else if (subdet == "HF") nChan=1728;

  ib.setCurrentFolder(subdir_+"digi_info/"+subdet);
  hist.shape = ib.book1D(subdet+" Digi Shape",subdet+" Digi Shape;Time Slice",10,-0.5,9.5);
  hist.shapeThresh = ib.book1D(subdet+" Digi Shape - over thresh",
				  subdet+" Digi Shape - over thresh passing trigger and HF HT cuts;Time slice",
				  10,-0.5,9.5);
  hist.ThreshCount = ib.book1D(subdet+" Total Digis Over Threshold",
				  subdet+" Total Digis Over Threshold",
				  1,-0.5,0.5);
  // Create plots of sums of adjacent time slices
  for (int ts=0;ts<9;++ts)
    {
      name<<subdet<<" Plus Time Slices "<<ts<<" and "<<ts+1;
      hist.TS_sum_plus.push_back(ib.book1D(name.str().c_str(),name.str().c_str(),50, 0., 50.));
      name.str("");
      name<<subdet<<" Minus Time Slices "<<ts<<" and "<<ts+1;
      hist.TS_sum_minus.push_back(ib.book1D(name.str().c_str(),name.str().c_str(),50, 0., 50.));
      name.str("");
    }
  hist.presample= ib.book1D(subdet+" Digi Presamples",subdet+" Digi Presamples",50,-0.5,49.5);
  hist.BQ = ib.book1D(subdet+" Bad Quality Digis",subdet+" Bad Quality Digis",nChan+1,-0.5,nChan+0.5);
  //(hist.BQ->getTH1F())->LabelsOption("v");
  hist.BQFrac = ib.book1D(subdet+" Bad Quality Digi Fraction",subdet+" Bad Quality Digi Fraction",DIGI_BQ_FRAC_NBINS,(0-0.5/(DIGI_BQ_FRAC_NBINS-1)),1+0.5/(DIGI_BQ_FRAC_NBINS-1));
  hist.DigiFirstCapID = ib.book1D(subdet+" Capid 1st Time Slice",subdet+" Capid for 1st Time Slice;CapId (T0)- 1st CapId (T0);# of Events",7,-3.5,3.5);

  hist.DVerr = ib.book1D(subdet+" Data Valid Err Bits",subdet+" QIE Data Valid Err Bits",4,-0.5,3.5);
  hist.DVerr ->setBinLabel(1,"Err=0, DV=0",1);
  hist.DVerr ->setBinLabel(2,"Err=0, DV=1",1);
  hist.DVerr ->setBinLabel(3,"Err=1, DV=0",1);
  hist.DVerr ->setBinLabel(4,"Err=1, DV=1",1);
  hist.CapID = ib.book1D(subdet+" CapID",subdet+" CapID",4,-0.5,3.5);
  hist.ADC = ib.book1D(subdet+" ADC count per time slice",subdet+" ADC count per time slice",200,-0.5,199.5);
  hist.ADCsum = ib.book1D(subdet+" ADC sum", subdet+" ADC sum",200,-0.5,199.5);
  hist.fibBCNOff = ib.book1D(subdet+" Fiber Orbit Message Idle BCN Offset", subdet+" Fiber Orbit Message Idle BCN Offset;Offset from Expected",
				 15, -7.5, 7.5);
}

void HcalDigiMonitor::analyze(edm::Event const&e, edm::EventSetup const&s)
{
  HcalBaseDQMonitor::analyze(e, s);

  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;

  // Get HLT trigger information for HF timing study
  passedMinBiasHLT_=false;

  /////////////////////////////////////////////////////////////////
  // check if detectors whether they were ON
  edm::Handle<DcsStatusCollection> dcsStatus;
  e.getByToken(dcsStatusToken_, dcsStatus);
  
  if (dcsStatus.isValid() && dcsStatus->size() != 0) 
    {      
      if ((*dcsStatus)[0].ready(DcsStatus::HBHEa) &&
	  (*dcsStatus)[0].ready(DcsStatus::HBHEb) &&   
	  (*dcsStatus)[0].ready(DcsStatus::HBHEc))
	{	
	  hbhedcsON = true;
	  if (debug_) std::cout << "hbhe on" << std::endl;
	} 
      else hbhedcsON = false;

      if ((*dcsStatus)[0].ready(DcsStatus::HF))
	{
	  hfdcsON = true;
	  if (debug_) std::cout << "hf on" << std::endl;
	} 
      else hfdcsON = false;
    }
  ///////////////////////////////////////////////////////////////

  edm::Handle<edm::TriggerResults> hltRes;
  if (!(e.getByToken(tok_trigger_,hltRes)))
    {
      if (debug_>0) edm::LogWarning("HcalDigiMonitor")<<" Could not get HLT results with tag "<<hltresultsLabel_<<std::endl;
    }
  else
    {
      const edm::TriggerNames & triggerNames = e.triggerNames(*hltRes);
      const unsigned int nTrig(triggerNames.size());
      for (unsigned int i=0;i<nTrig;++i){
	  // repeat for minbias triggers
	  for (unsigned int k=0;k<MinBiasHLTBits_.size();++k)
	    {
	      // if (triggerNames.triggerName(i)==MinBiasHLTBits_[k] && hltRes->accept(i))
	      if (triggerNames.triggerName(i).find(MinBiasHLTBits_[k])!=std::string::npos && hltRes->accept(i))
		{ 
		  passedMinBiasHLT_=true;
		  break;
		}
	    }
	}
    } //else
  
  // Now get collections we need
  HT_HFP_=0;
  HT_HFM_=0;
  //  bool rechitsFound=false;
  edm::Handle<HFRecHitCollection> hf_rechit;
  if (e.getByToken(tok_hfrec_,hf_rechit))
    {
      //      rechitsFound=true;
      for (HFRecHitCollection::const_iterator HF=hf_rechit->begin();HF!=hf_rechit->end();++HF)
	{
	  float en=HF->energy();
	  int ieta=HF->id().ieta();
	  // ieta for HF starts at 29, so subtract away 29 when computing fEta
	  double fEta=fabs(0.5*(theHFEtaBounds[abs(ieta)-28]+theHFEtaBounds[abs(ieta)-29]));
	  ieta>0 ?  HT_HFP_+=en/cosh(fEta) : HT_HFM_+=en/cosh(fEta);
	}
    }
  else
    {
      // if no rechits found, form above-threshold plots based only on digi comparison to ADC threshold 
      HT_HFP_=999;
      HT_HFM_=999;
    }

  // try to get digis
  edm::Handle<HBHEDigiCollection> hbhe_digi;
  edm::Handle<HODigiCollection> ho_digi;
  edm::Handle<HFDigiCollection> hf_digi;

  if (!(e.getByToken(tok_hbhe_,hbhe_digi)))
    {
      edm::LogWarning("HcalDigiMonitor")<< digiLabel_<<" hbhe_digi not available";
      return;
    }
  
  if (!(e.getByToken(tok_hf_,hf_digi)))
    {
      edm::LogWarning("HcalDigiMonitor")<< digiLabel_<<" hf_digi not available";
      return;
    }
  if (!(e.getByToken(tok_ho_,ho_digi)))
    {
      edm::LogWarning("HcalDigiMonitor")<< digiLabel_<<" ho_digi not available";
      return;
    }
  edm::Handle<HcalUnpackerReport> report;  
  if (!(e.getByToken(tok_unpack_,report)))
    {
      edm::LogWarning("HcalDigiMonitor")<< digiLabel_<<" unpacker report not available";
      return;
    }
  // try to get Raw Data
  edm::Handle<FEDRawDataCollection> rawraw;
  if ( !(e.getByToken(FEDRawDataCollectionToken_, rawraw)))
    {
      edm::LogWarning("HcalRawDataMonitor")<<" raw data with label "<<FEDRawDataCollection_<<" not available";
      return;
    }

  // get the DCC header & trailer (or bail out)
  // this needs to be done better, for now basically getting only one number per HBHE/HO/HF
  // will create a map (dccid, spigot) -> DetID to be used in process_Digi later
  for (int i=FEDNumbering::MINHCALFEDID; 
		  i<=FEDNumbering::MAXHCALuTCAFEDID; i++) {
	  if (i>FEDNumbering::MAXHCALFEDID && i<FEDNumbering::MINHCALuTCAFEDID)
		  continue;
    const FEDRawData& fed = rawraw->FEDData(i);
    if (fed.size()<12) continue;  //At least the size of headers and trailers of a DCC.    

    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
    if(!dccHeader) return;
	if (debug_>0)
		std::cout << "### Processing FED: " << i << std::endl;

	//	For uTCA spigos are useless => by default we have digisize = 4
	//	As of 20.05.2015
	//	HF = 2
	if ((i>=1118 && i<=1122) ||
			(i>=718 && i<=723))
	{
		mindigisizeHF_ = 4;
		maxdigisizeHF_ = 4;
		DigiExpectedSize->Fill(2, 4);
		continue;
	}

	//	VME readout contains Number of Time Samples per Digi
	//	uTCA doesn't!
    HcalHTRData htr;  
    for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
      if (!dccHeader->getSpigotPresent(spigot)) continue;
      
      // Load the given decoder with the pointer and length from this spigot.
      // i.e.     initialize htr, within dcc raw data size.
      dccHeader->getSpigotData(spigot, htr, fed.size()); 
      
      int NTS = htr.getNDD(); //number time slices, in precision channels
	  if (debug_>0)
		  std::cout << "### Number of TS=" << NTS << std::endl;
      if (NTS==0) continue; // no DAQ data in this HTR (fully zero-suppressed)
      int dccid=dccHeader->getSourceId();
      
      if(dccid==720 && (spigot==12 || spigot==13)) continue; // calibration HTR
      if(dccid==722 && (spigot==12 || spigot==13)) continue; // ZDC HTR

      int subdet = -1;
      
      if(dccid >= 700 && dccid<=717)  { subdet = 0; mindigisizeHBHE_ = NTS; maxdigisizeHBHE_ = NTS; } // HBHE
      if((dccid >= 1118 && dccid<=1122) || 
			  (dccid>=718 && dccid<=723))  
	  { subdet = 2; mindigisizeHF_ = NTS; maxdigisizeHF_ = NTS; }     // HF
      if(dccid >= 724 && dccid<=731)  { subdet = 1; mindigisizeHO_ = NTS; maxdigisizeHO_ = NTS; }     // HO
      
      DigiExpectedSize->Fill(subdet,int(NTS),1);
    }
  }

  // all objects grabbed; event is good
  if (debug_>1) std::cout <<"\t<HcalDigiMonitor::analyze>  Processing good event! event # = "<<ievt_<<std::endl;

//  HcalBaseDQMonitor::analyze(e,s); // base class increments ievt_, etc. counters

  // Digi collection was grabbed successfully; process the Event
  processEvent(*hbhe_digi, *ho_digi, *hf_digi, *conditions_,
	       *report, e.orbitNumber(),e.bunchCrossing());
  
} //void HcalDigiMonitor::analyze(...)

void HcalDigiMonitor::processEvent(const HBHEDigiCollection& hbhe,
				   const HODigiCollection& ho,
				   const HFDigiCollection& hf,
				   const HcalDbService& cond,
				   const HcalUnpackerReport& report,
				   int orN, int bcN)
{ 

  // Skip events in which minimal good digis found -- still getting some strange (calib?) events through DQM

  DigiUnpackerErrorCount->Fill(report.badQualityDigis());
  
  unsigned int allgooddigis= hbhe.size()+ho.size()+hf.size();

  // new data format in HCAL marks idle messages in the abort gap as bad capid. 
  // ignore this events. Also, sometimes there are many corrupted digis left 
  // from the QIE reset: ignore if in abort gap
  if(bcN>=3446 && bcN<=3564)
    if( (report.badQualityDigis()>100 && hbhe.size()==0) || (report.badQualityDigis()>1000) )
      return;

  // bad threshold:  ignore events in which bad outnumber good by more than 100:1
  // (one RBX in HBHE seems to send valid data occasionally even on QIE resets, which is why we can't just require allgooddigis==0 when looking for events to skip)    
  if ((allgooddigis==0) ||
      (1.*report.badQualityDigis()>100*allgooddigis))
    {
      h_valid_digis->Fill(1);
      if (bcN>-1)
	h_invalid_bcn->Fill(bcN);
      if (orN>-1)
	h_invalid_orbitnumMod103->Fill(orN%103);
      
      return;
    }
  
  h_valid_digis->Fill(0);

  // hbHists.count_bad=0;
  // hbHists.count_good=0;
  // heHists.count_bad=0;
  // heHists.count_good=0;
  // hoHists.count_bad=0;
  // hoHists.count_good=0;
  // hfHists.count_bad=0;
  // hfHists.count_good=0;

  // int HO0bad=0;
  // int HO12bad=0;
  // int HFlumibad=0;

  // Check unpacker report for bad digis

  typedef std::vector<DetId> DetIdVector;

  for ( DetIdVector::const_iterator baddigi_iter=report.bad_quality_begin(); 
	baddigi_iter != report.bad_quality_end();
	++baddigi_iter)
    {
      HcalDetId id(baddigi_iter->rawId());
      int rDepth = id.depth();
      int rPhi   = id.iphi();
      int rEta   = id.ieta();
      int binEta = CalcEtaBin(id.subdet(), rEta, rDepth); // why is this here?
      
      if (binEta < 85 && binEta >= 0 
	  && (rPhi-1) >= 0 && (rPhi-1)<72 
	  && (rDepth-1) >= 0 && (rDepth-1)<4)
	if(uniqcounter2[binEta][rPhi-1][rDepth-1]<1)  	
	  {
	    if (id.subdet()==HcalBarrel) ++hbHists.count_bad;	  
	    else if (id.subdet()==HcalEndcap) ++heHists.count_bad;
	    else if (id.subdet()==HcalForward) 
	      {
		++hfHists.count_bad;
		if (rDepth==1 && (abs(rEta)==33 || abs(rEta)==34)) ++HFlumibad;
		else if (rDepth==2 && (abs(rEta)==35 || abs(rEta)==36)) ++HFlumibad;
	      }
	    else if (id.subdet()==HcalOuter) 
	      {
		// Mark HORing+/-2 channels as present, HO/YB+/-2 has HV off (at 100V).
		if (excludeHORing2_==true && rDepth==4)
		  if (abs(rEta)>=11 && abs(rEta)<=15 && !isSiPM(rEta,rPhi,rDepth)) continue;
		
		if (excludeHO1P02_==true)
		  if( (rEta>4 && rEta<10) && (rPhi<=10 || rPhi>70) ) continue;
		
		if (KnownBadCells_.find(id)!=KnownBadCells_.end()) continue;
		
		++hoHists.count_bad;
		if (abs(rEta)<5) ++HO0bad;
		else ++HO12bad;
	      }
	    else 
	      continue; // skip anything that isn't HB, HE, HO, HF
	    // extra protection against nonsensical values -- prevents occasional crashes
	    
	    ++badunpackerreport[binEta][rPhi-1][rDepth-1];
	    ++baddigis[binEta][rPhi-1][rDepth-1];  
	    
	    // QPLL unlocking channels, have to ignore unpacker errors (fix requires opening CMS)
	    bool HEM15A = true ? (id.subdet()==HcalEndcap && (rPhi>56 && rPhi<59 && rEta<0)) : false;
	    bool HEM15B = true ? (id.subdet()==HcalEndcap && (rPhi>54 && rPhi<57 && rEta<0)) : false;
	    bool HBP14A = true ? (id.subdet()==HcalBarrel && (rPhi>50 && rPhi<53 && rEta>0)) : false;

	    if(excludeBadQPLLs_ && rDepth==1)
	      if( HEM15A || HEM15B || HBP14A )
		++knownbadQPLLs;
		
	    uniqcounter2[binEta][rPhi-1][rDepth-1]++;
	  }
    }
 ///////////////////////////////////////// Loop over HBHE

  int firsthbcap=-1;
  int firsthecap=-1;
  int firsthocap=-1;
  int firsthfcap=-1;
  
  for (HBHEDigiCollection::const_iterator j=hbhe.begin(); j!=hbhe.end(); ++j)
    {
	const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
	
	if (digi.id().subdet()==HcalBarrel)
	  {
	    if (!HBpresent_) continue;
	    if (KnownBadCells_.find(digi.id())!=KnownBadCells_.end()) continue;
	    
	    process_Digi(digi, hbHists, firsthbcap);

	  }
	else if (digi.id().subdet()==HcalEndcap)
	  {
	    if (!HEpresent_) continue;
	    process_Digi(digi, heHists,firsthecap);
	  }	
    }

  // // Fill good digis vs lumi block; also fill bad errors?
  HBocc_vs_LB->Fill(currentLS,hbHists.count_good);
  HEocc_vs_LB->Fill(currentLS,heHists.count_good);

  // Calculate number of bad quality cells and bad quality fraction
  if (HBpresent_ && (hbHists.count_good>0 || hbHists.count_bad>0))
    {
      int counter=hbHists.count_bad;
      if (counter<DIGI_SUBDET_NUM)
	++hbHists.count_BQ[counter];
      float counter2 = (1.*hbHists.count_bad)/(hbHists.count_bad+hbHists.count_good)*(DIGI_BQ_FRAC_NBINS-1);
      if (counter2<DIGI_SUBDET_NUM) ++hbHists.count_BQFrac[(int)counter2];
    }

  if (HEpresent_ && (heHists.count_good>0 || heHists.count_bad>0))
    {
      int counter=heHists.count_bad;
      if (counter<DIGI_SUBDET_NUM)
	++heHists.count_BQ[counter];
      float counter2 = (1.*heHists.count_bad)/(heHists.count_bad+heHists.count_good)*(DIGI_BQ_FRAC_NBINS-1);
      if (counter2<DIGI_SUBDET_NUM) ++heHists.count_BQFrac[int(counter2)];
    }

  //////////////////////////////////// Loop over HO collection
  if (HOpresent_)
    {
      for (HODigiCollection::const_iterator j=ho.begin(); j!=ho.end(); ++j)
	{
	  const HODataFrame digi = (const HODataFrame)(*j);
	  // Mark HORing+/-2 channels as present, HO/YB+/-2 has HV off (at 100V).
	  if (excludeHORing2_==true && digi.id().depth()==4)
	    if (abs(digi.id().ieta())>=11 && abs(digi.id().ieta())<=15 && 
		!isSiPM(digi.id().ieta(),digi.id().iphi(),digi.id().depth())) continue;
	  
	  if (excludeHO1P02_==true)	  
	    if( (digi.id().ieta()>4 && digi.id().ieta()<10) 
		&& (digi.id().iphi()<=10 || digi.id().iphi()>70) ) continue;

	  if (KnownBadCells_.find(digi.id())!=KnownBadCells_.end()) continue;
	  
	  process_Digi(digi, hoHists, firsthocap);
	} // for (HODigiCollection)
      
      if (hoHists.count_bad>0 || hoHists.count_good>0)
	{
	  int counter=hoHists.count_bad;
	  if (counter<DIGI_SUBDET_NUM)
	    ++hoHists.count_BQ[counter];

	  float counter2 = (1.*hoHists.count_bad)/(hoHists.count_bad+hoHists.count_good)*(DIGI_BQ_FRAC_NBINS-1);
	  if (counter2<DIGI_SUBDET_NUM) ++hoHists.count_BQFrac[int(counter2)];
	}
      HOocc_vs_LB->Fill(currentLS,hoHists.count_good);
    } // if (HOpresent_)
  
  /////////////////////////////////////// Loop over HF collection
  if (HFpresent_)
    {
      for (HFDigiCollection::const_iterator j=hf.begin(); j!=hf.end(); ++j)
	{
	  const HFDataFrame digi = (const HFDataFrame)(*j);
	  process_Digi(digi, hfHists, firsthfcap);
	} // for (HFDigiCollection)

      if (hfHists.count_bad>0 || hfHists.count_good>0)
	{
	  int counter=hfHists.count_bad;
	  if (counter<DIGI_SUBDET_NUM)
	    ++hfHists.count_BQ[counter];
	  float counter2 = (1.*hfHists.count_bad)/(hfHists.count_bad+hfHists.count_good)*(DIGI_BQ_FRAC_NBINS-1);
	  if (counter2<DIGI_SUBDET_NUM) ++hfHists.count_BQFrac[int(counter2)];
	}
      HFocc_vs_LB->Fill(currentLS,hfHists.count_good);
    } // if (HFpresent_)
  
  // This only counts digis that are present but bad somehow; it does not count digis that are missing
  int count_good=hbHists.count_good+heHists.count_good+hoHists.count_good+hfHists.count_good;
  int count_bad=hbHists.count_bad+heHists.count_bad+hoHists.count_bad+hfHists.count_bad;

  if (count_good<DIGI_NUM)
    ++diginum[count_good];

  // Fill bad quality histograms
  DigiUnpackerErrorFrac->Fill(1.*report.badQualityDigis()/(report.badQualityDigis()+count_good));
  DigiBQ->Fill(count_bad);
  if (count_bad>0 || count_good>0)
    DigiBQFrac->Fill(1.*count_bad/(count_bad+count_good));

  // Call 'update' on all histograms so that they update in online DQM  
  UpdateHists(hbHists);
  UpdateHists(heHists);
  UpdateHists(hoHists);
  UpdateHists(hfHists);

  // Now update global (non-subdetector-specific) histograms
  DigiNum->update();
  DigiErrorVME->update();
  DigiErrorSpigot->update();
  DigiBQ->update();
  DigiBQFrac->update();
  DigiUnpackerErrorCount->update();
  DigiUnpackerErrorFrac->update();

  // Update eta-phi hists
  for (unsigned int zz=0;zz<DigiErrorOccupancyByDepth.depth.size();++zz)
      DigiErrorOccupancyByDepth.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsByDepth.depth.size();++zz)
      DigiErrorsByDepth.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsBadCapID.depth.size();++zz)
      DigiErrorsBadCapID.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsDVErr.depth.size();++zz)
      DigiErrorsDVErr.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsBadDigiSize.depth.size();++zz)
      DigiErrorsBadDigiSize.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsBadADCSum.depth.size();++zz)
      DigiErrorsBadADCSum.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsUnpacker.depth.size();++zz)
      DigiErrorsUnpacker.depth[zz]->update();
  for (unsigned int zz=0;zz<DigiErrorsBadFibBCNOff.depth.size();++zz)
    DigiErrorsBadFibBCNOff.depth[zz]->update();

  DigiOccupancyEta->update();
  DigiOccupancyPhi->update();
  DigiOccupancyVME->update();
  DigiOccupancySpigot->update();
  DigiSize->update();
  DigiExpectedSize->update();

  // // Fill problems vs. lumi block plots
  // ProblemsVsLB->Fill(currentLS,count_bad);
  // ProblemsVsLB_HB->Fill(currentLS,hbHists.count_bad);
  // ProblemsVsLB_HE->Fill(currentLS,heHists.count_bad);
  // ProblemsVsLB_HO->Fill(currentLS,hoHists.count_bad);
  // ProblemsVsLB_HF->Fill(currentLS,hfHists.count_bad);
  // ProblemsVsLB_HBHEHF->Fill(currentLS,hbHists.count_bad+heHists.count_bad+hfHists.count_bad);

  // // Fill the number of problem digis in each channel
  // ProblemsCurrentLB->Fill(-1,-1,1);  // event counter
  // ProblemsCurrentLB->Fill(0,0,hbHists.count_bad);
  // ProblemsCurrentLB->Fill(1,0,heHists.count_bad);
  // ProblemsCurrentLB->Fill(2,0,hoHists.count_bad);
  // ProblemsCurrentLB->Fill(3,0,hfHists.count_bad);
  // ProblemsCurrentLB->Fill(4,0,HO0bad);
  // ProblemsCurrentLB->Fill(5,0,HO12bad);
  // ProblemsCurrentLB->Fill(6,0,HFlumibad);

  // Call fill method every checkNevents
  //fill_Nevents();
  
  return;
} // void HcalDigiMonitor::processEvent(...)



template <class DIGI>
int HcalDigiMonitor::process_Digi(DIGI& digi, DigiHists& h, int& firstcap)
{
  int err=0x0;
  bool bitUp = false;
  int ADCcount=0;

  int shapeThresh=0;
  
  int mindigisize=1;
  int maxdigisize=10;

  if (digi.id().subdet()==HcalBarrel)
    {
      shapeThresh=shapeThreshHB_;
      mindigisize=mindigisizeHBHE_;
      maxdigisize=maxdigisizeHBHE_;
    }
  else if (digi.id().subdet()==HcalEndcap)
    {
      shapeThresh=shapeThreshHE_;
      mindigisize=mindigisizeHBHE_;
      maxdigisize=maxdigisizeHBHE_;
    }
  else if (digi.id().subdet()==HcalOuter)
    {
      shapeThresh=shapeThreshHO_;
      mindigisize=mindigisizeHO_;
      maxdigisize=maxdigisizeHO_;
    }
  else if (digi.id().subdet()==HcalForward)
    {
      shapeThresh=shapeThreshHF_;
      mindigisize=mindigisizeHF_;
      maxdigisize=maxdigisizeHF_;
    }
  int iEta = digi.id().ieta();
  int iPhi = digi.id().iphi();
  int iDepth = digi.id().depth();
  int calcEta = CalcEtaBin(digi.id().subdet(),iEta,iDepth);  

  // Check that digi size is correct
  if (digi.size()<mindigisize || digi.size()>maxdigisize)
    {
      if (digi_checkdigisize_) err|=0x1;
      ++baddigisize[calcEta][iPhi-1][iDepth-1];
    }
  // Check digi size; if > 20, increment highest bin of digisize array
  if (digi.size()<20)
    ++digisize[static_cast<int>(digi.size())][digi.id().subdet()-1];
  else
    ++digisize[19][digi.id().subdet()-1];

  // loop over time slices of digi to check capID and errors
  ++h.count_presample[digi.presamples()];

  // Check CapID rotation
  if (firstcap==-1) firstcap = digi.sample(0).capid();
  int capdif = digi.sample(0).capid() - firstcap;
  //capdif = capdif%3 - capdif/3; // unnecessary?
  // capdif should run from -3 to +3
  if (capdif >-4 && capdif<4)
    ++h.capIDdiff[capdif+3];
  else
      ++h.capIDdiff[7];

  int last=-1;
  
  int offset = digi.fiberIdleOffset();

  // Only count BCN offset errors if ExpectedOrbitMessage Time is >-1
  // For offline (and thus cfg default), this won't be checked, since
  // we can't keep up to date with changes.
  if (offset != -1000 && DigiMonitor_ExpectedOrbitMessageTime_>-1) 
    {
      // increment counters only for non-zero offsets?
      ++h.fibbcnoff[offset + 7];
      if (offset != 0)
	{
	  ++badFibBCNOff[calcEta][iPhi-1][iDepth-1];
	  if (shutOffOrbitTest_ == false) err |= 0xF; // not an error if test turned off
	}
    }

  int tssum=0;

  bool digi_error=false;

  const int DigiSize=digi.size();
  for (int i=0;i<10;++i) pedSubtractedADC_[i]=0;
  const int pedSubADCsize=sizeof(pedSubtractedADC_)/sizeof(double);

  std::map<HcalDetId, std::vector<double> >::iterator  foundID = PedestalsByCapId_.find(digi.id());
  for (int i=0;i<DigiSize;++i)
    {
      int thisCapid = digi.sample(i).capid();
      if (thisCapid>=0 && thisCapid<4) ++h.capid[thisCapid];

      if (makeDiagnostics_)
	{
	  if(bitUpset(last,thisCapid)) bitUp=true; // checking capID rotation
	  last = thisCapid;
	  // Check for digi error bits
	  if (digi_checkdverr_)
	    {
	      if(digi.sample(i).er()) err=(err|0x2);
	      if(!digi.sample(i).dv()) err=(err|0x2);
	    }
	  if ((digi_error==false) && (digi.sample(i).er() || !digi.sample(i).dv()))
	    {
	      ++digierrorsdverr[calcEta][iPhi-1][iDepth-1];
	      digi_error=true; // only count 1 error per digi in this plot
	    }
	  ++h.dverr[static_cast<int>(2*digi.sample(i).er()+digi.sample(i).dv())];
	} // if (makeDiagnostics_)

      h.count_shape[i]+=digi.sample(i).adc();
      
      // Calculate ADC sum of adjacent samples -- still necessary?
      if (i==digi.size()-1) continue;
      tssum= digi.sample(i).adc()+digi.sample(i+1).adc();
      if (tssum<50 && tssum>=0)
	{
	  if (iEta>0)
	    ++h.tssumplus[tssum][i];
	  else
	    ++h.tssumminus[tssum][i];
	}

      if (digi.sample(i).adc()<0) ++h.adc[0];
      else if (digi.sample(i).adc()<200) ++h.adc[digi.sample(i).adc()];
      else ++h.adc[199];

      if (i>=pedSubADCsize) continue; // don't exceed maximum array length when checking digis
      
      if (foundID!=PedestalsByCapId_.end())
	{
	  pedSubtractedADC_[i]=digi.sample(i).adc()-(foundID->second)[thisCapid];
	  ADCcount+=(int)(digi.sample(i).adc()-(foundID->second)[thisCapid]);
	}
      else
	{
	  pedSubtractedADC_[i]=digi.sample(i).adc()-3;
	  ADCcount+=digi.sample(i).adc()-3; // default pedestal subtraction of 3 ADC counts
	}
    } // for (int i=0;i<digi.size();++i)
  
  // capid error found
  if(bitUp) 
    {
      if (digi_checkcapid_) err=(err|0x4);
      ++badcapID[calcEta][iPhi-1][iDepth-1];
    }


  // These plots generally don't get filled, unless we turn off the suppression of bad digis
  if (err>0)
    {
      if(uniqcounter[calcEta][iPhi-1][iDepth-1]<1)  
      	{
      	  ++h.count_bad;
      	  ++baddigis[calcEta][iPhi-1][iDepth-1];
      	  ++errorVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
      	  ++errorSpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];
      	}
      uniqcounter[calcEta][iPhi-1][iDepth-1]++; 

      return err;
    }

  if (ADCcount<0) ADCcount=0;
  if (ADCcount<199)
    ++h.adcsum[ADCcount];
  else
    ++h.adcsum[199]; // effective overflow bin

  // require larger threshold to look at pulse shapes
  
  if (ADCcount>shapeThresh && passedMinBiasHLT_  && HT_HFP_>1 && HT_HFM_>1)
    {
      h.ThreshCount->Fill(0,1);
      if (digi.id().subdet()!=HcalOuter || isSiPM(iEta,iPhi, iDepth)==false)
	{
	  for (int i=0;i<pedSubADCsize;++i)
	    h.count_shapeThresh[i]+=pedSubtractedADC_[i];
	}
    }

  // occupancy plots are only filled for good histograms
  ++h.count_good;
  ++occupancyEtaPhi[calcEta][iPhi-1][iDepth-1];
  ++occupancyEta[iEta+41];
  ++occupancyPhi[iPhi-1];
  // htr Slots run from 0-20, incremented by 0.5 for top/bottom
  ++occupancyVME[static_cast<int>(2*(digi.elecId().htrSlot()+0.5*digi.elecId().htrTopBottom()))][static_cast<int>(digi.elecId().readoutVMECrateId())];
  ++occupancySpigot[static_cast<int>(digi.elecId().spigot())][static_cast<int>(digi.elecId().dccid())];

  // Pawel's code for HF timing checks -- run only in online mode for non-calib events
  if (digi.id().subdet()==HcalForward 
      && Online_  //only run online 
      && currenttype_==0  // require non-calibration event
      && passedMinBiasHLT_ // require min bias trigger
      )
    {
      int maxtime=-1;
      double maxenergy=-1, fullenergy=0;
      int digisize=digi.size();
      for (int ff=0;ff<digisize;++ff)
	{
	  fullenergy+=digi.sample(ff).nominal_fC()-2.5;
	  if (digi.sample(ff).nominal_fC()-2.5>maxenergy)
	    {
	      maxenergy=digi.sample(ff).nominal_fC()-2.5;
	      maxtime=ff;
	    }
	}
    
      if (maxtime>=2 && maxtime<=5 && maxenergy>20 && maxenergy<100)  // only look between time slices 2-5; anything else should be nonsense
	{
	  for (int ff=0;ff<digisize;++ff){
	    if(fullenergy>0){
	      if(digi.id().ieta()>0)HFP_shape->Fill(ff,(digi.sample(ff).nominal_fC()-2.5)/fullenergy);
	      if(digi.id().ieta()<0)HFM_shape->Fill(ff,(digi.sample(ff).nominal_fC()-2.5)/fullenergy);
	    }
          }

	  double time_den=0, time_num=0;
	  // form weighted time sum
	  int startslice=std::max(0,maxtime-1);
	  int endslice=std::min(digisize-1,maxtime+1);
	  for (int ss=startslice;ss<=endslice;++ss)
	    {
	      // subtract 'default' pedestal of 2.5 fC
	      time_num+=ss*(digi.sample(ss).nominal_fC()-2.5);
	      time_den+=digi.sample(ss).nominal_fC()-2.5;
	    }

	  int myiphi=iPhi;
	  if (iDepth==2) ++myiphi;
	  if (HFtiming_etaProfile!=0 && time_den!=0)
	    HFtiming_etaProfile->Fill(iEta,time_num/time_den);
	  if (HFtiming_totaltime2D!=0 && time_den!=0)
	    HFtiming_totaltime2D->Fill(iEta,myiphi,time_num/time_den);
	  if (HFtiming_occupancy2D!=0 && time_den!=0)
	    HFtiming_occupancy2D->Fill(iEta,myiphi,1);
	} //maxtime>-1
    } // if HcalForward

  return err;
} // template <class DIGI> int HcalDigiMonitor::process_Digi

void HcalDigiMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					     const edm::EventSetup& c) 
{
  HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);
  ProblemsCurrentLB->Reset();
}

void HcalDigiMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					   const edm::EventSetup& c)
{
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;

  // Reset current LS histogram
  if (ProblemsCurrentLB)
    ProblemsCurrentLB->Reset();
  
  ProblemDigisInLastNLB_HBHEHF_alarm->Reset();

  //increase the number of LS counting, for alarmer. Only make alarms for HBHE
  if(hbhedcsON == true && hfdcsON == true && HBpresent_ == 1 && HEpresent_ == 1 && HFpresent_ == 1)
    ++alarmer_counter_; 
  else 
    alarmer_counter_ = 0;

  fill_Nevents();

  zeroCounters(); // reset counters of good/bad digis
 
  return;
}

void HcalDigiMonitor::fill_Nevents()
{
  if (debug_>0)
    std::cout <<"<HcalDigiMonitor> Calling fill_Nevents for event  "<<tevt_<< " (processed events = "<<ievt_<<")"<<std::endl;
  int iPhi, iEta, iDepth;
  //  bool valid=false;

  // Fill problems vs. lumi block plots
  ProblemsVsLB->Fill(currentLS,hbHists.count_bad+heHists.count_bad+hoHists.count_bad+hfHists.count_bad);
  ProblemsVsLB_HB->Fill(currentLS,hbHists.count_bad);
  ProblemsVsLB_HE->Fill(currentLS,heHists.count_bad);
  ProblemsVsLB_HO->Fill(currentLS,hoHists.count_bad);
  ProblemsVsLB_HF->Fill(currentLS,hfHists.count_bad);
  ProblemsVsLB_HBHEHF->Fill(currentLS,hbHists.count_bad+heHists.count_bad+hfHists.count_bad);

  if( hbHists.count_bad+heHists.count_bad+hfHists.count_bad-knownbadQPLLs< 50 )
    alarmer_counter_ = 0;
      
  if( alarmer_counter_ >= 5 )
    ProblemDigisInLastNLB_HBHEHF_alarm->Fill( std::min(int(hbHists.count_bad+heHists.count_bad+hfHists.count_bad), 99) );

  // Fill the number of problem digis in each channel
  if (ProblemsCurrentLB)
    {      
      ProblemsCurrentLB->Fill(-1,-1,1);  // event counter
      ProblemsCurrentLB->Fill(0,0,hbHists.count_bad);
      ProblemsCurrentLB->Fill(1,0,heHists.count_bad);
      ProblemsCurrentLB->Fill(2,0,hoHists.count_bad);
      ProblemsCurrentLB->Fill(3,0,hfHists.count_bad);
      ProblemsCurrentLB->Fill(4,0,HO0bad);
      ProblemsCurrentLB->Fill(5,0,HO12bad);
      ProblemsCurrentLB->Fill(6,0,HFlumibad);
    }

  // Fill plots of sums of adjacent digi samples
  for (int i=0;i<10;++i)
    {
      for (int j=0;j<50;++j)
	{
	  if (hbHists.tssumplus[j][i]>0) hbHists.TS_sum_plus[i]->Fill(j, hbHists.tssumplus[j][i]);
	  if (hbHists.tssumminus[j][i]>0) hbHists.TS_sum_minus[i]->Fill(j, hbHists.tssumminus[j][i]);	  
	  if (heHists.tssumplus[j][i]>0) heHists.TS_sum_plus[i]->Fill(j, heHists.tssumplus[j][i]); 
	  if (heHists.tssumminus[j][i]>0) heHists.TS_sum_minus[i]->Fill(j, heHists.tssumminus[j][i]); 
	  if (hoHists.tssumplus[j][i]>0) hoHists.TS_sum_plus[i]->Fill(j, hoHists.tssumplus[j][i]); 
	  if (hoHists.tssumminus[j][i]>0) hoHists.TS_sum_minus[i]->Fill(j, hoHists.tssumminus[j][i]); 
	  if (hfHists.tssumplus[j][i]>0) hfHists.TS_sum_plus[i]->Fill(j, hfHists.tssumplus[j][i]); 
	  if (hfHists.tssumminus[j][i]>0) hfHists.TS_sum_minus[i]->Fill(j, hfHists.tssumminus[j][i]); 
	}
    } // for (int i=0;i<10;++i)

  for (int i=0;i<DIGI_NUM;++i)
    {
      if (diginum[i]>0) DigiNum->Fill(i, diginum[i]);
      if (i>=DIGI_SUBDET_NUM) continue;

      if (hbHists.count_BQ[i]>0) hbHists.BQ->Fill(i, hbHists.count_BQ[i]);
      if (heHists.count_BQ[i]>0) heHists.BQ->Fill(i, heHists.count_BQ[i]);
      if (hoHists.count_BQ[i]>0) hoHists.BQ->Fill(i, hoHists.count_BQ[i]);
      if (hfHists.count_BQ[i]>0) hfHists.BQ->Fill(i, hfHists.count_BQ[i]);
    }//for int i=0;i<DIGI_NUM;++i)


  // Fill data-valid/error plots and capid plots
  for (int i=0;i<4;++i)
    {
      if (hbHists.dverr[i]>0) hbHists.DVerr->Fill(i, hbHists.dverr[i]);
      if (heHists.dverr[i]>0) heHists.DVerr->Fill(i, heHists.dverr[i]);
      if (hoHists.dverr[i]>0) hoHists.DVerr->Fill(i, hoHists.dverr[i]);
      if (hfHists.dverr[i]>0) hfHists.DVerr->Fill(i, hfHists.dverr[i]);

      if (hbHists.capid[i]>0) hbHists.CapID->Fill(i, hbHists.capid[i]);
      if (heHists.capid[i]>0) heHists.CapID->Fill(i, heHists.capid[i]);
      if (hoHists.capid[i]>0) hoHists.CapID->Fill(i, hoHists.capid[i]);
      if (hfHists.capid[i]>0) hfHists.CapID->Fill(i, hfHists.capid[i]);
    }

  for (int i=0;i<200;++i)
    {
      if (hbHists.adc[i]>0) hbHists.ADC->Fill(i, hbHists.adc[i]);

      if (heHists.adc[i]>0) heHists.ADC->Fill(i, heHists.adc[i]);
      if (hoHists.adc[i]>0) hoHists.ADC->Fill(i, hoHists.adc[i]);
      if (hfHists.adc[i]>0) hfHists.ADC->Fill(i, hfHists.adc[i]);

      if (hbHists.adcsum[i]>0) hbHists.ADCsum->Fill(i, hbHists.adcsum[i]);
      if (heHists.adcsum[i]>0) heHists.ADCsum->Fill(i, heHists.adcsum[i]);
      if (hoHists.adcsum[i]>0) hoHists.ADCsum->Fill(i, hoHists.adcsum[i]);
      if (hfHists.adcsum[i]>0) hfHists.ADCsum->Fill(i, hfHists.adcsum[i]);
    }

  for (int i = 0; i < 15; ++i)
    {
      if (hbHists.fibbcnoff[i]>0) hbHists.fibBCNOff->Fill(i-7, 
							  hbHists.fibbcnoff[i]);
      if (heHists.fibbcnoff[i]>0) heHists.fibBCNOff->Fill(i-7, 
							  heHists.fibbcnoff[i]);
      if (hfHists.fibbcnoff[i]>0) hfHists.fibBCNOff->Fill(i-7, 
							  hfHists.fibbcnoff[i]);
      if (hoHists.fibbcnoff[i]>0) hoHists.fibBCNOff->Fill(i-7,
							  hoHists.fibbcnoff[i]);
    }

  // Fill plots of bad fraction of digis found
  for (int i=0;i<DIGI_BQ_FRAC_NBINS;++i)
    {
      if (DIGI_BQ_FRAC_NBINS==1) break;
      if (hbHists.count_BQFrac[i]>0) hbHists.BQFrac->Fill(1.*i/(DIGI_BQ_FRAC_NBINS-1), hbHists.count_BQFrac[i]);
      if (heHists.count_BQFrac[i]>0) heHists.BQFrac->Fill(1.*i/(DIGI_BQ_FRAC_NBINS-1), heHists.count_BQFrac[i]);
      if (hoHists.count_BQFrac[i]>0) 
	{
	  hoHists.BQFrac->Fill(1.*i/(DIGI_BQ_FRAC_NBINS), hoHists.count_BQFrac[i]);
	}
      if (hfHists.count_BQFrac[i]>0) hfHists.BQFrac->Fill(1.*i/(DIGI_BQ_FRAC_NBINS-1), hfHists.count_BQFrac[i]);
    }//for (int i=0;i<DIGI_BQ_FRAC_NBINS;++i)

  // Fill presample plots
  for (int i=0;i<50;++i)
    {
      if (hbHists.count_presample[i]>0) hbHists.presample->Fill(i, hbHists.count_presample[i]);
      if (heHists.count_presample[i]>0) heHists.presample->Fill(i, heHists.count_presample[i]);
      if (hoHists.count_presample[i]>0) hoHists.presample->Fill(i, hoHists.count_presample[i]);
      if (hfHists.count_presample[i]>0) hfHists.presample->Fill(i, hfHists.count_presample[i]);
    } //for (int i=0;i<50;++i)

  // Fill shape plots
  for (int i=0;i<10;++i)
    {
      if (hbHists.count_shape[i]>0) hbHists.shape->Fill(i, hbHists.count_shape[i]);
      if (hbHists.count_shapeThresh[i]>0) hbHists.shapeThresh->Fill(i, hbHists.count_shapeThresh[i]);
      if (heHists.count_shape[i]>0) heHists.shape->Fill(i, heHists.count_shape[i]);
      if (heHists.count_shapeThresh[i]>0) heHists.shapeThresh->Fill(i, heHists.count_shapeThresh[i]);
      if (hoHists.count_shape[i]>0) hoHists.shape->Fill(i, hoHists.count_shape[i]);
      if (hoHists.count_shapeThresh[i]>0) hoHists.shapeThresh->Fill(i, hoHists.count_shapeThresh[i]);
      if (hfHists.count_shape[i]>0) hfHists.shape->Fill(i, hfHists.count_shape[i]);
      if (hfHists.count_shapeThresh[i]>0) hfHists.shapeThresh->Fill(i, hfHists.count_shapeThresh[i]);
    }//  for (int i=0;i<10;++i)

  // Fill capID difference plots
  for (int i=0;i<8;++i)
    {
      if (hbHists.capIDdiff[i]>0) hbHists.DigiFirstCapID->Fill(i, hbHists.capIDdiff[i]);
      if (heHists.capIDdiff[i]>0) heHists.DigiFirstCapID->Fill(i, heHists.capIDdiff[i]);
      if (hoHists.capIDdiff[i]>0) hoHists.DigiFirstCapID->Fill(i, hoHists.capIDdiff[i]);
      if (hfHists.capIDdiff[i]>0) hfHists.DigiFirstCapID->Fill(i, hfHists.capIDdiff[i]);
    }

  // Fill VME plots
  for (int i=0;i<40;++i)
    {
      for (int j=0;j<18;++j)
	{
	  if (errorVME[i][j]>0) DigiErrorVME->Fill(i, j,errorVME[i][j]);
	  if (occupancyVME[i][j]>0) DigiOccupancyVME->Fill(i, j,occupancyVME[i][j]);
	}
    } //for (int i=0;i<40;++i)

  // Fill SPIGOT plots
  for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)
    {
      for (int j=0;j<36;++j)
	{
	  if (errorSpigot[i][j]>0) DigiErrorSpigot->Fill(i, j,errorSpigot[i][j]);
	  if (occupancySpigot[i][j]>0) DigiOccupancySpigot->Fill(i, j,occupancySpigot[i][j]);
	}
    } //for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)

  // Loop over subdetectors
  for (int sub=0;sub<4;++sub)
    {
      for (int dsize=0;dsize<20;++dsize)
	{
	  if (digisize[dsize][sub]>0)
	    DigiSize->Fill(sub,dsize,digisize[dsize][sub]);
	}
    } // for (int sub=0;sub<4;++sub)

  // Loop over eta, phi, depth
  for (int d=0;d<4;++d)
    {
      iDepth=d+1;
      DigiErrorsByDepth.depth[d]->setBinContent(0,0,ievt_); // underflow bin contains event counter
      DigiOccupancyByDepth.depth[d]->setBinContent(0,0,ievt_);
      DigiErrorsBadDigiSize.depth[d]->setBinContent(0,0,ievt_);
      DigiErrorsUnpacker.depth[d]->setBinContent(0,0,ievt_);
      DigiErrorsBadFibBCNOff.depth[d]->setBinContent(0,0,ievt_);

      for (int phi=0;phi<72;++phi)
	{
	  iPhi=phi+1;
	  DigiOccupancyPhi->Fill(iPhi,occupancyPhi[phi]);
	  for (int eta=0;eta<83;++eta)
	    {
	      // DigiOccupanyEta uses 'true' ieta (included the overlap at +/- 29)
	      iEta=eta-41;
	      if (phi==0)
		DigiOccupancyEta->Fill(iEta,occupancyEta[eta]);
	      //	      valid=false;
	
	      // HB
	      if (validDetId(HcalBarrel, iEta, iPhi, iDepth))
		{
		  //		  valid=true;
		  if (HBpresent_)
		    {
                      int calcEta = CalcEtaBin(HcalBarrel,iEta,iDepth);

		      DigiOccupancyByDepth.depth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      if (makeDiagnostics_)
			{
			  DigiErrorsBadCapID.depth[d]->Fill(iEta, iPhi,
							    badcapID[calcEta][phi][d]);
			  DigiErrorsDVErr.depth[d]->Fill(iEta, iPhi,
							 digierrorsdverr[calcEta][phi][d]);
			}
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta, iPhi,
							   baddigisize[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							    badFibBCNOff[calcEta][phi][d]);
		      DigiErrorsUnpacker.depth[d]->Fill(iEta, iPhi,
							badunpackerreport[calcEta][phi][d]);
		      DigiErrorsByDepth.depth[d]->Fill(iEta, iPhi,
						       baddigis[calcEta][phi][d]);
		      // Use this for testing purposes only
		      //DigiErrorsByDepth[d]->Fill(iEta, iPhi, ievt_);
		    } // if (HBpresent_)
		} // validDetId(HB)
	      // HE
	      if (validDetId(HcalEndcap, iEta, iPhi, iDepth))
		{
		  //		  valid=true;
		  if (HEpresent_)
		    {
                      int calcEta = CalcEtaBin(HcalEndcap,iEta,iDepth);

		      DigiOccupancyByDepth.depth[d]->Fill(iEta, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      if (makeDiagnostics_)
			{
			  DigiErrorsBadCapID.depth[d]->Fill(iEta, iPhi,
							    badcapID[calcEta][phi][d]);
			  DigiErrorsDVErr.depth[d]->Fill(iEta, iPhi,
							 digierrorsdverr[calcEta][phi][d]);
			}
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta, iPhi,
							   baddigisize[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							    badFibBCNOff[calcEta][phi][d]);
		      DigiErrorsUnpacker.depth[d]->Fill(iEta, iPhi,
							badunpackerreport[calcEta][phi][d]);
		      DigiErrorsByDepth.depth[d]->Fill(iEta, iPhi,
						       baddigis[calcEta][phi][d]);
		    } // if (HEpresent_)
		} // valid HE found
	      // HO
	      if (validDetId(HcalOuter,iEta,iPhi,iDepth))
		{
		  //		  valid=true;
		  if (HOpresent_)
		    {
                      int calcEta = CalcEtaBin(HcalOuter,iEta,iDepth);
		      DigiOccupancyByDepth.depth[d]->Fill(iEta, iPhi,
							  occupancyEtaPhi[calcEta][phi][d]);
		      if (makeDiagnostics_)
			{
			  DigiErrorsBadCapID.depth[d]->Fill(iEta, iPhi,
							    badcapID[calcEta][phi][d]);
			  DigiErrorsDVErr.depth[d]->Fill(iEta, iPhi,
							 digierrorsdverr[calcEta][phi][d]);
			}
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta, iPhi,
							   baddigisize[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta, iPhi,
							    badFibBCNOff[calcEta][phi][d]);
		      DigiErrorsUnpacker.depth[d]->Fill(iEta, iPhi,
							badunpackerreport[calcEta][phi][d]);
		      
		      DigiErrorsByDepth.depth[d]->Fill(iEta,iPhi,
						       baddigis[calcEta][phi][d]);
		    } // if (HOpresent_)
		}//validDetId(HO)
	      // HF
	      if (validDetId(HcalForward,iEta,iPhi,iDepth))
		{
		  //		  valid=true;
		  if (HFpresent_)
		    {
                      int calcEta = CalcEtaBin(HcalForward,iEta,iDepth);
                      int zside = iEta/abs(iEta);
		      DigiOccupancyByDepth.depth[d]->Fill(iEta+zside, iPhi,
						    occupancyEtaPhi[calcEta][phi][d]);
		      
		      if (makeDiagnostics_)
			{
			  DigiErrorsBadCapID.depth[d]->Fill(iEta+zside, iPhi,
							    badcapID[calcEta][phi][d]);
			  DigiErrorsDVErr.depth[d]->Fill(iEta+zside, iPhi,
							 digierrorsdverr[calcEta][phi][d]);
			}
		      DigiErrorsBadDigiSize.depth[d]->Fill(iEta+zside, iPhi,
						     baddigisize[calcEta][phi][d]);
		      DigiErrorsBadFibBCNOff.depth[d]->Fill(iEta+zside, iPhi,
							 badFibBCNOff[calcEta][phi][d]);
		      DigiErrorsUnpacker.depth[d]->Fill(iEta+zside, iPhi,
							badunpackerreport[calcEta][phi][d]);
		      DigiErrorsByDepth.depth[d]->Fill(iEta+zside, iPhi,
						       baddigis[calcEta][phi][d]);
		      
		    } // if (HFpresent_)
		}
	    } // for (int eta=0;...)
	} // for (int phi=0;...)
    } // for (int d=0;...)

  // Now fill all the unphysical cell values
  FillUnphysicalHEHFBins(DigiErrorsByDepth);
  if (makeDiagnostics_)
    {
      FillUnphysicalHEHFBins(DigiErrorsBadCapID);
      FillUnphysicalHEHFBins(DigiErrorsDVErr);
    }
  FillUnphysicalHEHFBins(DigiErrorsBadDigiSize);
  FillUnphysicalHEHFBins(DigiOccupancyByDepth);
  FillUnphysicalHEHFBins(DigiErrorsBadFibBCNOff);
  FillUnphysicalHEHFBins(DigiErrorsUnpacker);

  //  zeroCounters(); // reset counters of good/bad digis
 
  return;
} // void HcalDigiMonitor::fill_Nevents()


void HcalDigiMonitor::zeroCounters()
{
  // Set all histogram counters back to 0
  // Call this after all every N evnets

  /******** Zero all counters *******/
  for (int d=0; d<DEPTHBINS; d++) {
    for (int eta=0; eta<ETABINS; eta++) {
      for (int phi=0; phi<PHIBINS; phi++){
	uniqcounter[eta][phi][d] = 0.0;
	uniqcounter2[eta][phi][d] = 0.0;
      }
    }
  }

  hbHists.count_bad=0;
  hbHists.count_good=0;
  heHists.count_bad=0;
  heHists.count_good=0;
  hoHists.count_bad=0;
  hoHists.count_good=0;
  hfHists.count_bad=0;
  hfHists.count_good=0;

  knownbadQPLLs = 0;

  HO0bad=0;
  HO12bad=0;
  HFlumibad=0;

  for (int i=0;i<85;++i)
    {
      occupancyEta[i]=0;
      if (i<72)
        occupancyPhi[i]=0;
      for (int j=0;j<72;++j)
        {
          for (int k=0;k<4;++k)
            {
              baddigis[i][j][k]=0;
              badcapID[i][j][k]=0;
              baddigisize[i][j][k]=0;
              occupancyEtaPhi[i][j][k]=0;
              digierrorsdverr[i][j][k]=0;
	      badFibBCNOff[i][j][k]=0;
	      badunpackerreport[i][j][k]=0;
            }
        } // for (int j=0;j<72;++i)
    } // for (int i=0;i<85;++i)

  for (int i=0;i<40;++i)
    {
      for (int j=0;j<18;++j)
	{
	  occupancyVME[i][j]=0;
	  errorVME[i][j]=0;
	}
    }

  for (int i=0;i<HcalDCCHeader::SPIGOT_COUNT;++i)
    {
      for (int j=0;j<36;++j)
	{
	  occupancySpigot[i][j]=0;
	  errorSpigot[i][j]=0;
	}
    }


  for (int i=0;i<20;++i)
    {
      for (int j=0;j<4;++j)
	digisize[i][j]=0;
    }

  for (int i=0;i<DIGI_NUM;++i)
    {
      diginum[i]=0;
      
      // set all DigiHists counters to 0
      if (i<4)
	{
	  hbHists.dverr[i]=0;
	  heHists.dverr[i]=0;
	  hoHists.dverr[i]=0;
	  hfHists.dverr[i]=0;
	  hbHists.capid[i]=0;
	  heHists.capid[i]=0;
	  hoHists.capid[i]=0;
	  hfHists.capid[i]=0;
	}
      if (i<8)
	{
	  hbHists.capIDdiff[i]=0;
	  heHists.capIDdiff[i]=0;
	  hoHists.capIDdiff[i]=0;
	  hfHists.capIDdiff[i]=0;
	}

      if (i<10)
	{
	  hbHists.count_shape[i]=0;
	  heHists.count_shape[i]=0;
	  hoHists.count_shape[i]=0;
	  hfHists.count_shape[i]=0;
	 
	  hbHists.count_shapeThresh[i]=0;
	  heHists.count_shapeThresh[i]=0;
	  hoHists.count_shapeThresh[i]=0;
	  hfHists.count_shapeThresh[i]=0;
	}
      if (i<50)
	{
	  hbHists.count_presample[i]=0;
	  heHists.count_presample[i]=0;
	  hoHists.count_presample[i]=0;
	  hfHists.count_presample[i]=0;
	  for (int j=0;j<10;++j)
	    {
	      hbHists.tssumplus[i][j]=0;
	      heHists.tssumplus[i][j]=0;
	      hoHists.tssumplus[i][j]=0;
	      hfHists.tssumplus[i][j]=0;
	      hbHists.tssumminus[i][j]=0;
	      heHists.tssumminus[i][j]=0;
	      hoHists.tssumminus[i][j]=0;
	      hfHists.tssumminus[i][j]=0;
	    }
	}

      if (i<15)
	{
	  hbHists.fibbcnoff[i]=0;
	  heHists.fibbcnoff[i]=0;
	  hoHists.fibbcnoff[i]=0;
	  hfHists.fibbcnoff[i]=0;
	}

      if (i<200)
	{
	  hbHists.adc[i]=0;
	  heHists.adc[i]=0;
	  hoHists.adc[i]=0;
	  hfHists.adc[i]=0;
	  hbHists.adcsum[i]=0;
	  heHists.adcsum[i]=0;
	  hoHists.adcsum[i]=0;
	  hfHists.adcsum[i]=0;
	}
      if (i<DIGI_SUBDET_NUM)
	{
	  hbHists.count_BQ[i]=0;
	  heHists.count_BQ[i]=0;
	  hoHists.count_BQ[i]=0;
	  hfHists.count_BQ[i]=0;
	}
      if (i<DIGI_BQ_FRAC_NBINS)
	{
	  hbHists.count_BQFrac[i]=0;
	  heHists.count_BQFrac[i]=0;
	  hoHists.count_BQFrac[i]=0;
	  hfHists.count_BQFrac[i]=0;
	}
    } // for (int i=0;i<DIGI_NUM;++i)


  return;
}

void HcalDigiMonitor::UpdateHists(DigiHists& h)
{
  // call update command for all histograms (should make them update when running in online DQM?)
  h.shape->update();
  h.shapeThresh->update();
  h.presample->update();
  h.BQ->update();
  h.BQFrac->update();
  h.DigiFirstCapID->update();
  h.DVerr->update();
  h.CapID->update();
  h.ADC->update();
  h.ADCsum->update();
  h.fibBCNOff->update();

  for (unsigned int i=0;i<h.TS_sum_plus.size();++i)
    h.TS_sum_plus[i]->update();
  for (unsigned int i=0;i<h.TS_sum_minus.size();++i)
    h.TS_sum_minus[i]->update();
} //void HcalDigiMonitor::UpdateHists(DigiHists& h)


void HcalDigiMonitor::reset()
{
  // reset the temporary histograms
  zeroCounters();
  
  // then reset the MonitorElements

  ProblemDigisInLastNLB_HBHEHF_alarm->Reset();
  alarmer_counter_ = 0;
  knownbadQPLLs    = 0; 

  hbhedcsON = true; hfdcsON = true;

  DigiErrorsByDepth.Reset();
  DigiErrorsBadCapID.Reset();
  DigiErrorsDVErr.Reset();
  DigiErrorsBadDigiSize.Reset();
  DigiErrorsBadADCSum.Reset();
  DigiErrorsUnpacker.Reset();
  DigiErrorsBadFibBCNOff.Reset();
  DigiOccupancyByDepth.Reset();
  DigiErrorOccupancyByDepth.Reset();

  DigiOccupancyEta->Reset();
  DigiOccupancyPhi->Reset();
  DigiOccupancyVME->Reset();
  DigiOccupancySpigot->Reset();
  DigiErrorVME->Reset();
  DigiErrorSpigot->Reset();
  
  DigiBQ->Reset();
  DigiBQFrac->Reset();
  DigiUnpackerErrorCount->Reset();
  DigiUnpackerErrorFrac->Reset();

  DigiNum->Reset();

  hbHists.shape->Reset();
  hbHists.shapeThresh->Reset();
  hbHists.presample->Reset();
  hbHists.BQ->Reset();
  hbHists.BQFrac->Reset();
  hbHists.DigiFirstCapID->Reset();
  hbHists.DVerr->Reset();
  hbHists.CapID->Reset();
  hbHists.ADC->Reset();
  hbHists.ADCsum->Reset();
  hbHists.fibBCNOff->Reset();
  for (unsigned int i=0;i<hbHists.TS_sum_plus.size();++i)
    hbHists.TS_sum_plus[i]->Reset();
  for (unsigned int i=0;i<hbHists.TS_sum_minus.size();++i)
    hbHists.TS_sum_minus[i]->Reset();

  heHists.shape->Reset();
  heHists.shapeThresh->Reset();
  heHists.presample->Reset();
  heHists.BQ->Reset();
  heHists.BQFrac->Reset();
  heHists.DigiFirstCapID->Reset();
  heHists.DVerr->Reset();
  heHists.CapID->Reset();
  heHists.ADC->Reset();
  heHists.ADCsum->Reset();
  heHists.fibBCNOff->Reset();
  for (unsigned int i=0;i<heHists.TS_sum_plus.size();++i)
    heHists.TS_sum_plus[i]->Reset();
  for (unsigned int i=0;i<heHists.TS_sum_minus.size();++i)
    heHists.TS_sum_minus[i]->Reset();

  hoHists.shape->Reset();
  hoHists.shapeThresh->Reset();
  hoHists.presample->Reset();
  hoHists.BQ->Reset();
  hoHists.BQFrac->Reset();
  hoHists.DigiFirstCapID->Reset();
  hoHists.DVerr->Reset();
  hoHists.CapID->Reset();
  hoHists.ADC->Reset();
  hoHists.ADCsum->Reset();
  hoHists.fibBCNOff->Reset();
  for (unsigned int i=0;i<hoHists.TS_sum_plus.size();++i)
    hoHists.TS_sum_plus[i]->Reset();
  for (unsigned int i=0;i<hoHists.TS_sum_minus.size();++i)
    hoHists.TS_sum_minus[i]->Reset();

  hfHists.shape->Reset();
  hfHists.shapeThresh->Reset();
  hfHists.presample->Reset();
  hfHists.BQ->Reset();
  hfHists.BQFrac->Reset();
  hfHists.DigiFirstCapID->Reset();
  hfHists.DVerr->Reset();
  hfHists.CapID->Reset();
  hfHists.ADC->Reset();
  hfHists.ADCsum->Reset();
  hfHists.fibBCNOff->Reset();
  for (unsigned int i=0;i<hfHists.TS_sum_plus.size();++i)
    hfHists.TS_sum_plus[i]->Reset();
  for (unsigned int i=0;i<hfHists.TS_sum_minus.size();++i)
    hfHists.TS_sum_minus[i]->Reset();

  return;
}
DEFINE_FWK_MODULE(HcalDigiMonitor);
               
