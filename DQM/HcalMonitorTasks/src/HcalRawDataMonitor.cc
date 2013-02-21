
#include "DQM/HcalMonitorTasks/interface/HcalRawDataMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"


HcalRawDataMonitor::HcalRawDataMonitor(const edm::ParameterSet& ps) {
  Online_                = ps.getParameter<bool>("online");
  mergeRuns_             = ps.getParameter<bool>("mergeRuns");
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup");
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder", "Hcal/"); // Hcal
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder", "RawDataMonitor_Hcal/"); // RawDataMonitor_Hcal
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",true);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",-1);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  FEDRawDataCollection_  = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection");
  digiLabel_             = ps.getUntrackedParameter<edm::InputTag>("digiLabel");

  excludeHORing2_       = ps.getUntrackedParameter<bool>("excludeHORing2",false);

  //inputLabelReport_      = ps.getUntrackedParameter<edm::InputTag>("UnpackerReport");
  
  // Initialize an array of MonitorElements
  meChann_DataIntegrityCheck_[0] =meCh_DataIntegrityFED00_;
  meChann_DataIntegrityCheck_[1] =meCh_DataIntegrityFED01_;
  meChann_DataIntegrityCheck_[2] =meCh_DataIntegrityFED02_;
  meChann_DataIntegrityCheck_[3] =meCh_DataIntegrityFED03_;
  meChann_DataIntegrityCheck_[4] =meCh_DataIntegrityFED04_;
  meChann_DataIntegrityCheck_[5] =meCh_DataIntegrityFED05_;
  meChann_DataIntegrityCheck_[6] =meCh_DataIntegrityFED06_;
  meChann_DataIntegrityCheck_[7] =meCh_DataIntegrityFED07_;
  meChann_DataIntegrityCheck_[8] =meCh_DataIntegrityFED08_;
  meChann_DataIntegrityCheck_[9] =meCh_DataIntegrityFED09_;
  meChann_DataIntegrityCheck_[10]=meCh_DataIntegrityFED10_;
  meChann_DataIntegrityCheck_[11]=meCh_DataIntegrityFED11_;
  meChann_DataIntegrityCheck_[12]=meCh_DataIntegrityFED12_;
  meChann_DataIntegrityCheck_[13]=meCh_DataIntegrityFED13_;
  meChann_DataIntegrityCheck_[14]=meCh_DataIntegrityFED14_;
  meChann_DataIntegrityCheck_[15]=meCh_DataIntegrityFED15_;
  meChann_DataIntegrityCheck_[16]=meCh_DataIntegrityFED16_;
  meChann_DataIntegrityCheck_[17]=meCh_DataIntegrityFED17_;
  meChann_DataIntegrityCheck_[18]=meCh_DataIntegrityFED18_;
  meChann_DataIntegrityCheck_[19]=meCh_DataIntegrityFED19_;
  meChann_DataIntegrityCheck_[20]=meCh_DataIntegrityFED20_;
  meChann_DataIntegrityCheck_[21]=meCh_DataIntegrityFED21_;
  meChann_DataIntegrityCheck_[22]=meCh_DataIntegrityFED22_;
  meChann_DataIntegrityCheck_[23]=meCh_DataIntegrityFED23_;
  meChann_DataIntegrityCheck_[24]=meCh_DataIntegrityFED24_;
  meChann_DataIntegrityCheck_[25]=meCh_DataIntegrityFED25_;
  meChann_DataIntegrityCheck_[26]=meCh_DataIntegrityFED26_;
  meChann_DataIntegrityCheck_[27]=meCh_DataIntegrityFED27_;
  meChann_DataIntegrityCheck_[28]=meCh_DataIntegrityFED28_;
  meChann_DataIntegrityCheck_[29]=meCh_DataIntegrityFED29_;
  meChann_DataIntegrityCheck_[30]=meCh_DataIntegrityFED30_;
  meChann_DataIntegrityCheck_[31]=meCh_DataIntegrityFED31_;
  
  this->reset();
} // HcalRawDataMonitor::HcalRawDataMonitor()

// destructor
HcalRawDataMonitor::~HcalRawDataMonitor(){}

// reset
void HcalRawDataMonitor::reset(void)
{

  for (int f=0; f<NUMDCCS; f++) {
    for (int s=0; s<15; s++) {
      UScount[f][s]=0;}}

  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      HalfHTRDataCorruptionIndicators_  [x][y]=0;

  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      LRBDataCorruptionIndicators_  [x][y]=0;
  	 
  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<TWO__SPGT; y++)
      ChannSumm_DataIntegrityCheck_[x][y]=0;

  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      DataFlowInd_[x][y]=0;

  for (int f=0; f<NUMDCCS; f++)
    for (int x=0; x<  TWO_CHANN; x++)
      for (int y=0; y<TWO__SPGT; y++)      
	Chann_DataIntegrityCheck_  [f][x][y]=0;

  for (int i=0; i<(NUMDCCS * NUMSPIGS * HTRCHANMAX); i++) 
    hashedHcalDetId_[i]=HcalDetId::Undefined;

  for (int d=0; d<DEPTHBINS; d++) {
    for (int eta=0; eta<ETABINS; eta++) {
      for (int phi=0; phi<PHIBINS; phi++){
	uniqcounter[eta][phi][d] = 0.0;
	problemcount[eta][phi][d] = 0.0;
	problemfound[eta][phi][d] = false;
      }
    }
  }

  // Properly initialze bylumi counters.
  NumBadHB=0;
  NumBadHE=0;
  NumBadHO=0;
  NumBadHF=0;
  NumBadHFLUMI=0;
  NumBadHO0=0;
  NumBadHO12=0;

} // HcalRawDataMonitor::HcalRawDataMonitor()

// BeginRun
void HcalRawDataMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c){
  HcalBaseDQMonitor::beginRun(run,c);
  edm::ESHandle<HcalDbService> pSetup;
  c.get<HcalDbRecord>().get( pSetup );

  readoutMap_=pSetup->getHcalMapping();
  DetId detid_;
  HcalDetId hcaldetid_; 

  // Build a map of readout hardware unit to calorimeter channel
  std::vector <HcalElectronicsId> AllElIds = readoutMap_->allElectronicsIdPrecision();
  uint32_t itsdcc    =0;
  uint32_t itsspigot =0;
  uint32_t itshtrchan=0;
  
  // by looping over all precision (non-trigger) items.
  for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin();
       eid != AllElIds.end();
       eid++) {

    //Get the HcalDetId from the HcalElectronicsId
    detid_ = readoutMap_->lookup(*eid);
    // NULL if illegal; ignore
    if (!detid_.null()) {
      if (detid_.det()!=4) continue; //not Hcal
      if (detid_.subdetId()!=HcalBarrel &&
	  detid_.subdetId()!=HcalEndcap &&
	  detid_.subdetId()!=HcalOuter  &&
	  detid_.subdetId()!=HcalForward) continue;

      itsdcc    =(uint32_t) eid->dccid(); 
      itsspigot =(uint32_t) eid->spigot();
      itshtrchan=(uint32_t) eid->htrChanId();
      hcaldetid_ = HcalDetId(detid_);
      stashHDI(hashup(itsdcc,itsspigot,itshtrchan),
	       hcaldetid_);
    } // if (!detid_.null()) 
  } 
}

// Begin LumiBlock
void HcalRawDataMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					      const edm::EventSetup& c) {
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  HcalBaseDQMonitor::beginLuminosityBlock(lumiSeg,c);
  //zeroCounters(); // zero hot cell counters at the start of each luminosity block
  ProblemsCurrentLB->Reset();
  return;
}
// Setup
void HcalRawDataMonitor::setup(void){
  // Call base class setup
  HcalBaseDQMonitor::setup();
  if (!dbe_) {
    if (debug_>1)
      std::cout <<"<HcalRawDataMonitor::setup>  No DQMStore instance available. Bailing out."<<std::endl;
    return;}

  /******* Set up all histograms  ********/
  if (debug_>1)
    std::cout <<"<HcalRawDataMonitor::beginRun>  Setting up histograms"<<std::endl;
  
  dbe_->setCurrentFolder(subdir_);
  ProblemsVsLB=dbe_->bookProfile("RAW_Problems_HCAL_vs_LS",
				 "Total HCAL RAW Problems vs lumi section", 
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,10000);

  ProblemsVsLB_HB=dbe_->bookProfile("Total_RAW_Problems_HB_vs_LS",
				    "Total HB RAW Problems vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);
  ProblemsVsLB_HE=dbe_->bookProfile("Total_RAW_Problems_HE_vs_LS",
				    "Total HE RAW Problems vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);
  ProblemsVsLB_HO=dbe_->bookProfile("Total_RAW_Problems_HO_vs_LS",
				    "Total HO RAW Problems vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,3000);
  ProblemsVsLB_HF=dbe_->bookProfile("Total_RAW_Problems_HF_vs_LS",
				    "Total HF RAW Problems vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,2000);
  ProblemsVsLB_HBHEHF=dbe_->bookProfile("Total_RAW_Problems_HBHEHF_vs_LS",
				    "Total HBHEHF RAW Problems vs lumi section",
				    NLumiBlocks_,0.5,NLumiBlocks_+0.5,100,0,2000);
 
  ProblemsVsLB->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HB->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HE->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HO->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HF->getTProfile()->SetMarkerStyle(20);
  ProblemsVsLB_HBHEHF->getTProfile()->SetMarkerStyle(20);
  MonitorElement* excludeHO2=dbe_->bookInt("ExcludeHOring2");
  // Fill with 0 if ring is not to be excluded; fill with 1 if it is to be excluded
  if (excludeHO2) excludeHO2->Fill(excludeHORing2_==true ? 1 : 0);


  //  Already done in base class:
  //dbe_->setCurrentFolder(subdir_);
  //meIevt_ = dbe_->bookInt("EventsProcessed");
  //if (meIevt_) meIevt_->Fill(-1);
  //meLevt_ = dbe_->bookInt("EventsProcessed_currentLS");
  //if (meLevt_) meLevt_->Fill(-1);
  //meTevt_ = dbe_->bookInt("EventsProcessed_All");
  //if (meTevt_) meTevt_->Fill(-1);
  //meTevtHist_=dbe_->book1D("EventsProcessed_AllHists","Counter of Events Processed By This Task",1,0.5,1.5);
  //if (meTevtHist_) meTevtHist_->Reset();
  
  std::string type;
      
  dbe_->setCurrentFolder(subdir_ + "Corruption"); /// Below, "Corruption" FOLDER
  type = "01 Common Data Format violations";
  meCDFErrorFound_ = dbe_->book2D(type,type,32,699.5,731.5,9,0.5,9.5);
  meCDFErrorFound_->setAxisTitle("HCAL FED ID", 1);
  meCDFErrorFound_->setBinLabel(1, "Hdr1BitUnset", 2);
  meCDFErrorFound_->setBinLabel(2, "FmtNumChange", 2);
  meCDFErrorFound_->setBinLabel(3, "BOE not '0x5'", 2);
  meCDFErrorFound_->setBinLabel(4, "Hdr2Bit Set", 2);
  meCDFErrorFound_->setBinLabel(5, "Hdr1 36-55", 2);
  meCDFErrorFound_->setBinLabel(6, "BOE not 0", 2);
  meCDFErrorFound_->setBinLabel(7, "Trlr1Bit Set", 2);
  meCDFErrorFound_->setBinLabel(8, "Size Error", 2);
  meCDFErrorFound_->setBinLabel(9, "TrailerBad", 2);
      
  type = "02 DCC Event Format violation";
  meDCCEventFormatError_ = dbe_->book2D(type,type,32,699.5,731.5,6,0.5,6.5);
  meDCCEventFormatError_->setAxisTitle("HCAL FED ID", 1);
  meDCCEventFormatError_->setBinLabel(1, "FmtVers Changed", 2);
  meDCCEventFormatError_->setBinLabel(2, "StrayBits Changed", 2);
  meDCCEventFormatError_->setBinLabel(3, "HTRStatusPad", 2);
  meDCCEventFormatError_->setBinLabel(4, "32bitPadErr", 2);
  meDCCEventFormatError_->setBinLabel(5, "Number Mismatch Bit Miscalc", 2);      
  meDCCEventFormatError_->setBinLabel(6, "Low 8 HTR Status Bits Miscopy", 2);	       
      
  type = "04 HTR BCN when OrN Diff";
  meBCNwhenOrNDiff_ = dbe_->book1D(type,type,3564,-0.5,3563.5);
  meBCNwhenOrNDiff_->setAxisTitle("BCN",1);
  meBCNwhenOrNDiff_->setAxisTitle("# of Entries",2);
      
  type = "03 OrN NonZero Difference HTR - DCC";
  meOrNCheck_ = dbe_->book1D(type,type,65,-32.5,32.5);
  meOrNCheck_->setAxisTitle("htr OrN - dcc OrN",1);
      
  type = "03 OrN Inconsistent - HTR vs DCC";
  meOrNSynch_= dbe_->book2D(type,type,32,700,732, 15,0,15);
  meOrNSynch_->setAxisTitle("FED #",1);
  meOrNSynch_->setAxisTitle("Spigot #",2);
      
  type = "05 BCN NonZero Difference HTR - DCC";
  meBCNCheck_ = dbe_->book1D(type,type,501,-250.5,250.5);
  meBCNCheck_->setAxisTitle("htr BCN - dcc BCN",1);
      
  type = "05 BCN Inconsistent - HTR vs DCC";
  meBCNSynch_= dbe_->book2D(type,type,32,700,732, 15,0,15);
  meBCNSynch_->setAxisTitle("FED #",1);
  meBCNSynch_->setAxisTitle("Slot #",2);
      
  type = "06 EvN NonZero Difference HTR - DCC";
  meEvtNCheck_ = dbe_->book1D(type,type,601,-300.5,300.5);
  meEvtNCheck_->setAxisTitle("htr Evt # - dcc Evt #",1);
      
  type = "06 EvN Inconsistent - HTR vs DCC";
  meEvtNumberSynch_= dbe_->book2D(type,type,32,700,732, 15,0,15);
  meEvtNumberSynch_->setAxisTitle("FED #",1);
  meEvtNumberSynch_->setAxisTitle("Slot #",2);
      
  //     ---------------- 
  //     | E!P | UE | TR |                                           
  // ----|  ND | OV | ID |					       
  // | T | CRC | ST | ODD| 					       
  // -------------------- 					       
  type="07 LRB Data Corruption Indicators";  
  meLRBDataCorruptionIndicators_= dbe_->book2D(type,type,
						THREE_FED,0,THREE_FED,
						THREE_SPG,0,THREE_SPG);
  label_xFEDs   (meLRBDataCorruptionIndicators_, 4); // 3 bins + 1 margin per ch.
  label_ySpigots(meLRBDataCorruptionIndicators_, 4); // 3 bins + 1 margin each spgt
      
  //     ---------------- 
  //     | CT | BE |    |
  //     | HM | 15 | WW | (Wrong Wordcount)
  //     | TM | CK | IW | (Illegal Wordcount)
  //     ---------------- 
  type="08 Half-HTR Data Corruption Indicators";
  meHalfHTRDataCorruptionIndicators_= dbe_->book2D(type,type,
						    THREE_FED,0,THREE_FED,
						    THREE_SPG,0,THREE_SPG);
  label_xFEDs   (meHalfHTRDataCorruptionIndicators_, 4); // 3 bins + 1 margin per ch.
  label_ySpigots(meHalfHTRDataCorruptionIndicators_, 4); // 3 bins + 1 margin each spgt
      
  //    ------------
  //    | !DV | Er  |
  //    | NTS | Cap |
  //    ------------
  type = "09 Channel Integrity Summarized by Spigot";
  meChannSumm_DataIntegrityCheck_= dbe_->book2D(type,type,
						 TWO___FED,0,TWO___FED,
						 TWO__SPGT,0,TWO__SPGT);
  label_xFEDs   (meChannSumm_DataIntegrityCheck_, 3); // 2 bins + 1 margin per ch.
  label_ySpigots(meChannSumm_DataIntegrityCheck_, 3); // 2 bins + 1 margin per spgt
      
  dbe_->setCurrentFolder(subdir_ + "Corruption/Channel Data Integrity");
  char label[256];
  for (int f=0; f<NUMDCCS; f++){      
    snprintf(label, 256, "FED %03d Channel Integrity", f+700);
    meChann_DataIntegrityCheck_[f] =  dbe_->book2D(label,label,
						    TWO_CHANN,0,TWO_CHANN,
						    TWO__SPGT,0,TWO__SPGT);
    label_xChanns (meChann_DataIntegrityCheck_[f], 3); // 2 bins + 1 margin per ch.
    label_ySpigots(meChann_DataIntegrityCheck_[f], 3); // 2 bins + 1 margin per spgt
    ;}
      
  dbe_->setCurrentFolder(subdir_ + "Data Flow"); ////Below, "Data Flow" FOLDER
  type="DCC Event Counts";
  mefedEntries_ = dbe_->book1D(type,type,32,699.5,731.5);
      
  type = "BCN from DCCs";
  medccBCN_ = dbe_->book1D(type,type,3564,-0.5,3563.5);
  medccBCN_->setAxisTitle("BCN",1);
  medccBCN_->setAxisTitle("# of Entries",2);
      
  type = "BCN from HTRs";
  meBCN_ = dbe_->book1D(type,type,3564,-0.5,3563.5);
  meBCN_->setAxisTitle("BCN",1);
  meBCN_->setAxisTitle("# of Entries",2);
      
  type = "DCC Data Block Size Distribution";
  meFEDRawDataSizes_=dbe_->book1D(type,type,1200,-0.5,12000.5);
  meFEDRawDataSizes_->setAxisTitle("# of bytes",1);
  meFEDRawDataSizes_->setAxisTitle("# of Data Blocks",2);
      
  type = "DCC Data Block Size Profile";
  meEvFragSize_ = dbe_->bookProfile(type,type,32,699.5,731.5,100,-1000.0,12000.0,"");
  type = "DCC Data Block Size Each FED";
  meEvFragSize2_ =  dbe_->book2D(type,type,64,699.5,731.5, 240,0,12000);
      
  //     ------------
  //     | OW | OFW |    "Two Caps HTR; Three Caps FED."
  //     | BZ | BSY |
  //     | EE | RL  |
  // ----------------
  // | CE |            (corrected error, Hamming code)
  // ------
  type = "01 Data Flow Indicators";
  meDataFlowInd_= dbe_->book2D(type,type,
				TWO___FED,0,TWO___FED,
				THREE_SPG,0,THREE_SPG);
  label_xFEDs   (meDataFlowInd_, 3); // 2 bins + 1 margin per ch.
  label_ySpigots(meDataFlowInd_, 4); // 3 bins + 1 margin each spgt
      
  dbe_->setCurrentFolder(subdir_ + "Diagnostics"); ////Below, "Diagnostics" FOLDER

  type = "DCC Firmware Version";
  meDCCVersion_ = dbe_->bookProfile(type,type, 32, 699.5, 731.5, 256, -0.5, 255.5);
  meDCCVersion_ ->setAxisTitle("FED ID", 1);
      
  type = "HTR Status Word HBHE";
  HTR_StatusWd_HBHE =  dbe_->book1D(type,type,16,-0.5,15.5);
  labelHTRBits(HTR_StatusWd_HBHE,1);
      
  type = "HTR Status Word HF";
  HTR_StatusWd_HF =  dbe_->book1D(type,type,16,-0.5,15.5);
  labelHTRBits(HTR_StatusWd_HF,1);
      
  type = "HTR Status Word HO";
  HTR_StatusWd_HO = dbe_->book1D(type,type,16,-0.5,15.5);
  labelHTRBits(HTR_StatusWd_HO,1);
      
  int maxbits = 16;//Look at all 16 bits of the Error Words
  type = "HTR Status Word by Crate";
  meStatusWdCrate_ = dbe_->book2D(type,type,18,-0.5,17.5,maxbits,-0.5,maxbits-0.5);
  meStatusWdCrate_ -> setAxisTitle("Crate #",1);
  labelHTRBits(meStatusWdCrate_,2);
      
  type = "Unpacking - HcalHTRData check failures";
  meInvHTRData_= dbe_->book2D(type,type,16,-0.5,15.5,32,699.5,731.5);
  meInvHTRData_->setAxisTitle("Spigot #",1);
  meInvHTRData_->setAxisTitle("DCC #",2);
      
  type = "HTR Fiber Orbit Message BCN";
  meFibBCN_ = dbe_->book1D(type,type,3564,-0.5,3563.5);
  meFibBCN_->setAxisTitle("BCN of Fib Orb Msg",1);
      
  type = "HTR Status Word - Crate 0";
  meCrate0HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate0HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate0HTRStatus_,2);
      
  type = "HTR Status Word - Crate 1";
  meCrate1HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate1HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate1HTRStatus_,2);
      
  type = "HTR Status Word - Crate 2";
  meCrate2HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate2HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate2HTRStatus_,2);
      
  type = "HTR Status Word - Crate 3";
  meCrate3HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate3HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate3HTRStatus_,2);
      
  type = "HTR Status Word - Crate 4";
  meCrate4HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate4HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate4HTRStatus_,2);

  type = "HTR Status Word - Crate 5";
  meCrate5HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate5HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate5HTRStatus_,2);

  type = "HTR Status Word - Crate 6";
  meCrate6HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate6HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate6HTRStatus_,2);

  type = "HTR Status Word - Crate 7";
  meCrate7HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate7HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate7HTRStatus_,2);

  type = "HTR Status Word - Crate 9";
  meCrate9HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate9HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate9HTRStatus_,2);

  type = "HTR Status Word - Crate 10";
  meCrate10HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate10HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate10HTRStatus_,2);

  type = "HTR Status Word - Crate 11";
  meCrate11HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate11HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate11HTRStatus_,2);

  type = "HTR Status Word - Crate 12";
  meCrate12HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate12HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate12HTRStatus_,2);

  type = "HTR Status Word - Crate 13";
  meCrate13HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate13HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate13HTRStatus_,2);

  type = "HTR Status Word - Crate 14";
  meCrate14HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate14HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate14HTRStatus_,2);

  type = "HTR Status Word - Crate 15";
  meCrate15HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate15HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate15HTRStatus_,2);

  type = "HTR Status Word - Crate 17";
  meCrate17HTRStatus_ = dbe_->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
  meCrate17HTRStatus_ ->setAxisTitle("Slot #",1);
  labelHTRBits(meCrate17HTRStatus_,2);

  type = "HTR UnSuppressed Event Fractions";
  meUSFractSpigs_ = dbe_->book1D(type,type,481,0,481);
  for(int f=0; f<NUMDCCS; f++) {
    snprintf(label, 256, "FED 7%02d", f);
    meUSFractSpigs_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
    for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
      snprintf(label, 256, "sp%02d", s);
      meUSFractSpigs_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

  // Firmware version
  type = "HTR Firmware Version";
  //  Maybe change to Profile histo eventually
  //meHTRFWVersion_ = dbe_->bookProfile(type,type,18,-0.5,17.5,245,10.0,255.0,"");
  meHTRFWVersion_ = dbe_->book2D(type,type ,18,-0.5,17.5,180,75.5,255.5);
  meHTRFWVersion_->setAxisTitle("Crate #",1);
  meHTRFWVersion_->setAxisTitle("HTR Firmware Version",2);

  type = "HTR Fiber 1 Orbit Message BCNs";
  meFib1OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 2 Orbit Message BCNs";
  meFib2OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 3 Orbit Message BCNs";
  meFib3OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 4 Orbit Message BCNs";
  meFib4OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 5 Orbit Message BCNs";
  meFib5OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 6 Orbit Message BCNs";
  meFib6OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 7 Orbit Message BCNs";
  meFib7OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
  type = "HTR Fiber 8 Orbit Message BCNs";
  meFib8OrbMsgBCN_= dbe_->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);

}

// Analyze
void HcalRawDataMonitor::analyze(const edm::Event& e, const edm::EventSetup& s){
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;

  HcalBaseDQMonitor::analyze(e,s); // base class increments ievt_, etc. counters
  
  // try to get die Data
  edm::Handle<FEDRawDataCollection> rawraw;
  if (!(e.getByLabel(FEDRawDataCollection_,rawraw)))
  //  if (!(e.getByType(rawraw)))
    {
      edm::LogWarning("HcalRawDataMonitor")<<" raw data with label "<<FEDRawDataCollection_ <<" not available";
      return;
    }
  edm::Handle<HcalUnpackerReport> report;  
  if (!(e.getByLabel(digiLabel_,report)))
    {
      edm::LogWarning("HcalRawDataMonitor")<<" Unpacker Report "<<digiLabel_<<" not available";
      return;
    }

  // all objects grabbed; event is good
  if (debug_>1) std::cout <<"\t<HcalRawDataMonitor::analyze>  Processing good event! event # = "<<ievt_<<std::endl;
  
  // Raw Data collection was grabbed successfully; process the Event
  processEvent(*rawraw, *report);

  //Loop over all cells, tallying problem counts before resetting
  //Extract the subdetector region given the location 
  for (int d=0; d<DEPTHBINS; d++) {
    for (int eta=0; eta<ETABINS; eta++) {
      int ieta=CalcIeta(eta,d+1);
      if (ieta==-9999) continue;
      for (int phi=0; phi<PHIBINS; phi++){
	if (problemcount[eta][phi][d]) {
	  //	  std::cout<<" "<<eta<<","<<phi<<","<<d<<" count:"<<problemcount[eta][phi][d]<<std::endl;	    
	  HcalSubdetector subdet=HcalEmpty;
	  if (isHB(eta,d+1))subdet=HcalBarrel;
	  else if (isHE(eta,d+1)) subdet=HcalEndcap;
	  else if (isHF(eta,d+1)) subdet=HcalForward;
	  else if (isHO(eta,d+1)) subdet=HcalOuter;
	  if (subdet!=HcalEmpty){
	    if (subdet==HcalBarrel)       {if(uniqcounter[eta][phi][d]<1) NumBadHB+= problemcount[eta][phi][d]; uniqcounter[eta][phi][d]++; }
	    else if (subdet==HcalEndcap)  {if(uniqcounter[eta][phi][d]<1) NumBadHE+= problemcount[eta][phi][d]; uniqcounter[eta][phi][d]++; }
	    ///NumBadHE+= problemcount[eta][phi][d];
	    else if (subdet==HcalOuter)  
	      {
		if(uniqcounter[eta][phi][d]<1) 
		  NumBadHO += problemcount[eta][phi][d];
		uniqcounter[eta][phi][d]++; 
		if (abs(ieta)<5) NumBadHO0+= problemcount[eta][phi][d];
		else NumBadHO12+= problemcount[eta][phi][d];
	      }
	    else if (subdet==HcalForward)
	      {
		if(uniqcounter[eta][phi][d]<1) 
		  NumBadHF+= problemcount[eta][phi][d];
		uniqcounter[eta][phi][d]++; 
		if (d==1 && (abs(ieta)==33 || abs(ieta)==34))
		  NumBadHFLUMI+= problemcount[eta][phi][d];
		else if (d==2 && (abs(ieta)==35 || abs(ieta)==36))
		  NumBadHFLUMI+= problemcount[eta][phi][d];
	      }
	  }
	  problemcount[eta][phi][d]=0;
	}
      }
    }
  }
}

void HcalRawDataMonitor::processEvent(const FEDRawDataCollection& rawraw, 
				      const HcalUnpackerReport& report){
  if(!dbe_) { 
    if (debug_>1)
      printf("HcalRawDataMonitor::processEvent DQMStore not instantiated!\n");  
    return;}
  
  // Fill event counters (underflow bins of histograms)
  meLRBDataCorruptionIndicators_->update();
  meHalfHTRDataCorruptionIndicators_->update();
  meChannSumm_DataIntegrityCheck_->update();
  for (int f=0; f<NUMDCCS; f++)      
    meChann_DataIntegrityCheck_[f]->update();
  meDataFlowInd_->update();

  // Loop over all FEDs reporting the event, unpacking if good.
  for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++) {
    const FEDRawData& fed = rawraw.FEDData(i);
    if (fed.size()<12) continue;  //At least the size of headers and trailers of a DCC.
    unpack(fed); //Interpret data, fill histograms, everything.
  }
  
  //increment problemcount[] where problemfound[], and clear problemfound[]
  for (int x=0; x<ETABINS; x++)
      for (int y=0; y<PHIBINS; y++)
	for (int z=0; z<DEPTHBINS; z++) 
	  if (problemfound[x][y][z]) {
	    problemcount[x][y][z]++;
	    problemfound[x][y][z]=false;
	  }
  return;
} //void HcalRawDataMonitor::processEvent()

// Process one FED's worth (one DCC's worth) of the event data.
void HcalRawDataMonitor::unpack(const FEDRawData& raw){

  // get the DCC header & trailer (or bail out)
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  // FED id declared in the header
  int dccid=dccHeader->getSourceId();
  //Force 0<= dcc_ <= 31
  int dcc_=std::max(0,dccid-700);  
  dcc_ = std::min(dcc_,31);       
  if(debug_>1) std::cout << "DCC " << dccid << std::endl;
  uint64_t* dccfw= (uint64_t*) (raw.data()+(sizeof(uint64_t)*2)); //64-bit DAQ word number 2 (from 0)
  int dcc_fw =  ( ((*dccfw)>>(6*8))&0x00000000000000FF );         //Shift right 6 bytes, get that low byte.
  meDCCVersion_ -> Fill(dccid,dcc_fw);

  //Before all else, how much data are we dealing with here?
  uint64_t* lastDataWord = (uint64_t*) ( raw.data()+(raw.size())-(1*sizeof(uint64_t)) );
  int EvFragLength = (int) (*lastDataWord>>(4*8)) & 0x0000000000FFFFFF ; //Shift right 4 bytes, get low 3 bytes.
  meFEDRawDataSizes_->Fill(EvFragLength*8);      //# 64-bit DAQ words *8 = # bytes. 
  meEvFragSize_ ->Fill(dccid, EvFragLength*8);   //# 64-bit DAQ words *8 = # bytes. 
  meEvFragSize2_ ->Fill(dccid, EvFragLength*8);  //# 64-bit DAQ words *8 = # bytes. 
  
  //Orbit, BunchCount, and Event Numbers
  unsigned long dccEvtNum = dccHeader->getDCCEventNumber();
  int dccBCN = dccHeader->getBunchId();
  //Mask down to 5 bits, since only used for comparison to HTR's five bit number...
  unsigned int dccOrN = (unsigned int) (dccHeader->getOrbitNumber() & 0x0000001F);
  medccBCN_ -> Fill(dccBCN);

  ////////// Histogram problems with the Common Data Format compliance;////////////
  bool CDFProbThisDCC = false; 
  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()
      //TESTME_HCALRAWDATA//
      //|| ((dccid==702)&&(tevt_%2==0))
      ) {
    meCDFErrorFound_->Fill(dccid, 1);
    CDFProbThisDCC = true; 
  }
  /* 2 */ //Make sure a reference CDF Version value has been recorded for this dccid
  CDFvers_it = CDFversionNumber_list.find(dccid);
  if (CDFvers_it  == CDFversionNumber_list.end()) {
    CDFversionNumber_list.insert(std::pair<int,short>
				 (dccid,dccHeader->getCDFversionNumber() ) );
    CDFvers_it = CDFversionNumber_list.find(dccid);
  } // then check against it.
  if (dccHeader->getCDFversionNumber()!= CDFvers_it->second) {
    meCDFErrorFound_->Fill(dccid,2);
    CDFProbThisDCC = true; 
  }
  /* 3 */ //There should always be a '5' in CDF Header word 0, bits [63:60]
  if (dccHeader->BOEshouldBe5Always()!=5) {
    meCDFErrorFound_->Fill(dccid, 3);
    CDFProbThisDCC = true; 
  }
  /* 4 */ //There should never be a third CDF Header word indicated.
  if (dccHeader->thereIsAThirdCDFHeaderWord()) {
    meCDFErrorFound_->Fill(dccid, 4);
    CDFProbThisDCC = true; 
  }
  /* 5 */ //Make sure a reference value of Reserved Bits has been recorded for this dccid
  CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  if (CDFReservedBits_it  == CDFReservedBits_list.end()) {
    CDFReservedBits_list.insert(std::pair<int,short>
 				(dccid,dccHeader->getSlink64ReservedBits() & 0x0000FFFF ) );
    CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  } // then check against it.
  if (((int) dccHeader->getSlink64ReservedBits() & 0x0000FFFF ) != CDFReservedBits_it->second) {
    meCDFErrorFound_->Fill(dccid,5);
    //CDFProbThisDCC = true; 
  }
  /* 6 */ //There should always be 0x0 in CDF Header word 1, bits [63:60]
  if (dccHeader->BOEshouldBeZeroAlways() !=0) {
    meCDFErrorFound_->Fill(dccid, 6);
    CDFProbThisDCC = true; 
  }
  /* 7 */ //There should only be one trailer
  if (trailer.moreTrailers()) {
    meCDFErrorFound_->Fill(dccid, 7);
    CDFProbThisDCC = true; 
  }
  //  if trailer.
  /* 8 */ //CDF Trailer [55:30] should be the # 64-bit words in the EvFragment
  if ((uint64_t) raw.size() != ( (uint64_t) trailer.lenght()*sizeof(uint64_t)) )  //The function name is a typo! Awesome.
    {
      meCDFErrorFound_->Fill(dccid, 8);
      CDFProbThisDCC = true; 
    }
  /* 9 */ //There is a rudimentary sanity check built into the FEDTrailer class
  if (!trailer.check()) {
    meCDFErrorFound_->Fill(dccid, 9);
    CDFProbThisDCC = true; 
  }
  if (CDFProbThisDCC) {
    //Set the problem flag for the ieta, iphi of any channel in this DCC
    mapDCCproblem(dcc_);
    if (debug_>0) std::cout <<"CDFProbThisDCC"<<std::endl;
  }

  mefedEntries_->Fill(dccid);

  CDFProbThisDCC = false;  // reset for the next go-round.
  
  char CRC_err;
  for(int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    CRC_err = ((dccHeader->getSpigotSummary(i) >> 10) & 0x00000001);
    if (CRC_err) {
      mapDCCproblem(dcc_);
      //Set the problem flag for the ieta, iphi of any channel in this DCC
      if (debug_>0) std::cout <<"HTR Problem: CRC_err"<<std::endl;
    }
  }
  
  // The DCC TTS state at event-sending time
  char TTS_state = (char)trailer.ttsBits();
  // The DCC TTS state at time L1A received (event enqueued to be built)
  char L1AtimeTTS_state=(char) dccHeader->getAcceptTimeTTS();
  if (TTS_state==L1AtimeTTS_state) ;//party

  ////////// Histogram problems with DCC Event Format compliance;////////////
  /* 1 */ //Make sure a reference value of the DCC Event Format version has been noted for this dcc.
  DCCEvtFormat_it = DCCEvtFormat_list.find(dccid);
  if (DCCEvtFormat_it == DCCEvtFormat_list.end()) {
    DCCEvtFormat_list.insert(std::pair<int,short>
			     (dccid,dccHeader->getDCCDataFormatVersion() ) );
    DCCEvtFormat_it = DCCEvtFormat_list.find(dccid);
  } // then check against it.
  if (dccHeader->getDCCDataFormatVersion()!= DCCEvtFormat_it->second) {
    meDCCEventFormatError_->Fill(dccid,1);
    mapDCCproblem(dcc_);
    if (debug_>0)std::cout <<"DCC Error Type 1"<<std::endl;
  }
  /* 2 */ //Check for ones where there should always be zeros
  if (false) //dccHeader->getByte1Zeroes() || dccHeader->getByte3Zeroes() || dccHeader->getByte567Zeroes()) 
  {
    meDCCEventFormatError_->Fill(dccid,2);
    mapDCCproblem(dcc_);
    if (debug_>0)std::cout <<"DCC Error Type 2"<<std::endl;
  }
  /* 3 */ //Check that there are zeros following the HTR Status words.
  int SpigotPad = HcalDCCHeader::SPIGOT_COUNT;
  if (  (((uint64_t) dccHeader->getSpigotSummary(SpigotPad)  ) 
	 | ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+1)) 
	 | ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+2)))  != 0){
    meDCCEventFormatError_->Fill(dccid,3);
    mapDCCproblem(dcc_);
  if (debug_>0)std::cout <<"DCC Error Type 3"<<std::endl;
  }
  /* 4 */ //Check that there are zeros following the HTR Payloads, if needed.
  int nHTR32BitWords=0;
  // add up all the declared HTR Payload lengths
  for(int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    nHTR32BitWords += dccHeader->getSpigotDataLength(i);  }
  // if it's an odd number, check for the padding zeroes
  if (( nHTR32BitWords % 2) == 1) {
    uint64_t* lastDataWord = (uint64_t*) ( raw.data()+raw.size()-(2*sizeof(uint64_t)) );
    if ((*lastDataWord>>32) != 0x00000000){
      meDCCEventFormatError_->Fill(dccid, 4);
      mapDCCproblem(dcc_);
      if (debug_>0)std::cout <<"DCC Error Type 4"<<std::endl;
    }
  }
  
  //unsigned char HTRErrorList=0; 
  //for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
  //  HTRErrorList=dccHeader->getSpigotErrorBits(j);    
  //}

  // These will be used in FED-vs-spigot 2D Histograms
  const int fed3offset = 1 + (4*dcc_); //3 bins, plus one of margin, each DCC
  const int fed2offset = 1 + (3*dcc_); //2 bins, plus one of margin, each DCC
  if (TTS_state & 0x8) /*RDY*/ 
    ;
  if (TTS_state & 0x2) /*SYN*/ 
    {
      mapDCCproblem(dcc_);
      if (debug_>0)std::cout <<"TTS_state Error:sync"<<std::endl;
    }
  //Histogram per-Spigot bits from the DCC Header
  int WholeErrorList=0; 
  for(int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
    if (!( dccHeader->getSpigotEnabled((unsigned int) spigot)) )
      continue; //skip when not enabled
    // This will be used in FED-vs-spigot 2D Histograms
    const int spg3offset = 1 + (4*spigot); //3 bins, plus one of margin, each spigot
    
    if (TTS_state & 0x4) /*BSY*/ 
      ++DataFlowInd_[fed2offset+1][spg3offset+1];
    if (TTS_state & 0x1) /*OFW*/ 
      ++DataFlowInd_[fed2offset+1][spg3offset+2];

    WholeErrorList=dccHeader->getLRBErrorBits((unsigned int) spigot);
    if (WholeErrorList!=0) {
      if ((WholeErrorList>>0)&0x01)  //HammingCode Corrected -- Not data corruption!
	DataFlowInd_[fed2offset-1][spg3offset-1]++;
      if (((WholeErrorList>>1)&0x01)!=0)  {//HammingCode Uncorrected Error
	LRBDataCorruptionIndicators_[fed3offset+1][spg3offset+2]++;
	mapHTRproblem(dcc_,spigot);
      }
      if (((WholeErrorList>>2)&0x01)!=0)  {//Truncated data coming into LRB
	LRBDataCorruptionIndicators_[fed3offset+2][spg3offset+2]++;
	mapHTRproblem(dcc_,spigot);
      }
      if (((WholeErrorList>>3)&0x01)!=0)  {//FIFO Overflow
	LRBDataCorruptionIndicators_[fed3offset+1][spg3offset+1]++;
	mapHTRproblem(dcc_,spigot);
      }
      if (((WholeErrorList>>4)&0x01)!=0)  {//ID (EvN Mismatch), htr payload metadeta
	LRBDataCorruptionIndicators_[fed3offset+2][spg3offset+1]++;
	mapHTRproblem(dcc_,spigot);
      }
      if (((WholeErrorList>>5)&0x01)!=0)  {//STatus: hdr/data/trlr error
	LRBDataCorruptionIndicators_[fed3offset+1][spg3offset+0]++;
	mapHTRproblem(dcc_,spigot);
      }
      if (((WholeErrorList>>6)&0x01)!=0)  {//ODD 16-bit word count from HTR
	LRBDataCorruptionIndicators_[fed3offset+2][spg3offset+0]++;
	mapHTRproblem(dcc_,spigot);
      }
    }
    if (!dccHeader->getSpigotPresent((unsigned int) spigot)){
      LRBDataCorruptionIndicators_[fed3offset+0][spg3offset+2]++;  //Enabled, but data not present!
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"HTR Problem: Spigot Not Present"<<std::endl;
    } else {
      if ( dccHeader->getSpigotDataTruncated((unsigned int) spigot)) {
     	LRBDataCorruptionIndicators_[fed3offset-1][spg3offset+0]++;  // EventBuilder truncated babbling LRB
	mapHTRproblem(dcc_,spigot);
	if (debug_>0)std::cout <<"HTR Problem: Spigot Data Truncated"<<std::endl;
      }
      if ( dccHeader->getSpigotCRCError((unsigned int) spigot)) {
	LRBDataCorruptionIndicators_[fed3offset+0][spg3offset+0]++; 
	mapHTRproblem(dcc_,spigot);
      }
    } //else spigot marked "Present"
    if (dccHeader->getSpigotDataLength(spigot) <(unsigned long)4) {
      LRBDataCorruptionIndicators_[fed3offset+0][spg3offset+1]++;  //Lost HTR Data for sure.
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"HTR Problem: Spigot Data Length too small"<<std::endl;
    }    
  }

  // Walk through the HTR data...
  HcalHTRData htr;  
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
    const int spg3offset = 1 + (4*spigot); //3 bins, plus one of margin, each spigot
    const int spg2offset = 1 + (3*spigot); //3 bins, plus one of margin, each spigot
    if (!dccHeader->getSpigotPresent(spigot)) continue;

    // Load the given decoder with the pointer and length from this spigot.
    // i.e.     initialize htr, within dcc raw data size.
    dccHeader->getSpigotData(spigot,htr, raw.size()); 
    const unsigned short* HTRraw = htr.getRawData();

    // check min length, correct wordcount, empty event, or total length if histo event.
    if (!htr.check()) {
      meInvHTRData_ -> Fill(spigot,dccid);
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"HTR Problem: HTR check fails"<<std::endl;
    }

    unsigned short HTRwdcount = htr.getRawLength();

    // Size checks for internal consistency
    // getNTP(), get NDD() seems to be mismatched with format. Manually:
    int NTP = ((htr.getExtHdr6() >> 8) & 0x00FF);
    int NDAQ = (HTRraw[htr.getRawLength() - 4] & 0x7FF);

    if ( !  ((HTRwdcount != 8)               ||
	     (HTRwdcount != 12 + NTP + NDAQ) ||
	     (HTRwdcount != 20 + NTP + NDAQ)    )) {
      ++HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+0];
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"HTR Problem: NTP+NDAQ size consistency check fails"<<std::endl;
      //incompatible Sizes declared. Skip it.
      continue; }
    bool EE = ((dccHeader->getSpigotErrorBits(spigot) >> 2) & 0x01);
    if (EE) { 
      if (HTRwdcount != 8) {	//incompatible Sizes declared. Skip it.
	++HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+1];
	if (debug_>0)std::cout <<"HTR Problem: HTRwdcount !=8"<<std::endl;	
      }
      DataFlowInd_[fed2offset+0][spg3offset+0]++;
      continue;}
    else{ //For non-EE, both CompactMode and !CompactMode
      bool CM = (htr.getExtHdr7() >> 14)&0x0001;
      int paddingsize = ((NDAQ+NTP)%2); //Extra padding to make HTRwdcount even
      if (( CM && ( (HTRwdcount-NDAQ-NTP-paddingsize) != 12) )
	  ||                                
	  (!CM && ( (HTRwdcount-NDAQ-NTP-paddingsize) != 20) )  ) {	//incompatible Sizes declared. Skip it.
	++HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+1];
	mapHTRproblem(dcc_,spigot);
	continue;} }

    if (htr.isHistogramEvent()) continue;

    //We trust the data now.  Finish with the check against DCCHeader
    unsigned int htrOrN = htr.getOrbitNumber(); 
    unsigned int htrBCN = htr.getBunchNumber(); 
    unsigned int htrEvtN = htr.getL1ANumber();
    meBCN_->Fill(htrBCN);  //The only periodic number for whole events.

    if (( (htrOrN  == dccOrN ) &&
	  (htrBCN  == (unsigned int) dccBCN) )  
	!= (dccHeader->getBxMismatchWithDCC(spigot))  ){
      meDCCEventFormatError_->Fill(dccid,5);
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"Orbit or BCN  HTR/DCC mismatch"<<std::endl;
    }
    if ( (htrEvtN == dccEvtNum) != 
	 dccHeader->getSpigotValid(spigot) ) {
      meDCCEventFormatError_->Fill(dccid,5);
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"DCC invalid spigot"<<std::endl;
    }
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    if (debug_ > 0) HTRPrint(htr,debug_);
    unsigned int htrFWVer = htr.getFirmwareRevision() & 0xFF;
    meHTRFWVersion_->Fill(cratenum,htrFWVer);  

    ///check that all HTRs have the same L1A number.
    int EvtNdiff = htrEvtN - dccEvtNum;
    if (EvtNdiff!=0) {
      meEvtNumberSynch_->Fill(dccid,spigot);
      mapHTRproblem(dcc_,spigot);
      meEvtNCheck_->Fill(EvtNdiff);
      if (debug_ == 1)std::cout << "++++ Evt # out of sync, ref, this HTR: "<< dccEvtNum << "  "<<htrEvtN <<std::endl;
    }

    ///check that all HTRs have the same BCN
    int BCNdiff = htrBCN-dccBCN;
    if ((BCNdiff!=0) 
	//TESTME_HCALRAWDATA//
	//|| ((dccid==727) && (spigot==8) && (dccEvtNum%3==0))
	){
      meBCNSynch_->Fill(dccid,spigot);
      mapHTRproblem(dcc_,spigot);
      meBCNCheck_->Fill(BCNdiff);
      if (debug_==1)std::cout << "++++ BCN # out of sync, ref, this HTR: "<< dccBCN << "  "<<htrBCN <<std::endl;
    }

    ///check that all HTRs have the same OrN
    int OrNdiff = htrOrN-dccOrN;
    if (OrNdiff!=0) {
      meOrNSynch_->Fill(dccid,spigot);
      mapHTRproblem(dcc_,spigot);
      meOrNCheck_->Fill(OrNdiff);
      meBCNwhenOrNDiff_->Fill(htrBCN); // Are there special BCN where OrN mismatched occur? Let's see.
      if (debug_==1)std::cout << "++++ OrN # out of sync, ref, this HTR: "<< dccOrN << "  "<<htrOrN <<std::endl;
    }

    bool htrUnSuppressed=(HTRraw[6]>>15 & 0x0001);
    if (htrUnSuppressed) {
      UScount[dcc_][spigot]++;
      int here=1+(HcalDCCHeader::SPIGOT_COUNT*(dcc_))+spigot;
      meUSFractSpigs_->Fill(here,
			    ((double)UScount[dcc_][spigot]));}

    MonitorElement* tmpErr = 0;
    HcalDetId HDI = hashedHcalDetId_[hashup(dcc_,spigot)];
    if (HDI != HcalDetId::Undefined) {
      switch (HDI.subdetId()) {
      case (HcalBarrel): {
	tmpErr = HTR_StatusWd_HBHE;
      } break;
      case (HcalEndcap): {
	tmpErr = HTR_StatusWd_HBHE;
      } break;
      case (HcalOuter): {
	tmpErr = HTR_StatusWd_HO;
      } break;
      case (HcalForward): {
	tmpErr = HTR_StatusWd_HF; 
      } break;
      default: break;
      }
    }
   
    int errWord = htr.getErrorsWord() & 0xFFFF;
    if (  (((dccHeader->getSpigotSummary( spigot))>>24)&0x00FF)
	  != (errWord&0x00FF) ){
      meDCCEventFormatError_->Fill(dccid,6);//Low 8 bits miscopied into DCCHeader
      mapHTRproblem(dcc_,spigot);
      if (debug_>0)std::cout <<"DCC spigot summary error or HTR error word"<<std::endl;
      //What other problems may lurk? Spooky.
    }
    if(tmpErr!=NULL){
      for(int i=0; i<16; i++){
	int errbit = errWord&(0x01<<i);
	// Bit 15 should always be 1; consider it an error if it isn't.
	if (i==15) errbit = errbit - 0x8000;
	if (errbit !=0) {
	  tmpErr->Fill(i);
	  //Only certain bits indicate corrupted data:
	  switch (i) {
	  case (14): //CT (obsolete Calibration Trigger scheme used)
	    HalfHTRDataCorruptionIndicators_[fed3offset+0][spg3offset+2]++;
	    mapHTRproblem(dcc_,spigot);
	    if (debug_>0)std::cout <<"HTR Problem: Case 14"<<std::endl;
	    break;
	  case (13): //HM
	    HalfHTRDataCorruptionIndicators_[fed3offset+0][spg3offset+1]++;
	    mapHTRproblem(dcc_,spigot);
	    if (debug_>0)std::cout <<"HTR Problem: Case 13"<<std::endl;
	    break;
	  case (12): //TM
	    HalfHTRDataCorruptionIndicators_[fed3offset+0][spg3offset+0]++;
	    mapHTRproblem(dcc_,spigot);
	    if (debug_>0)std::cout <<"HTR Problem: Case 12"<<std::endl;
	    break;
	  case ( 8): //BE
	    HalfHTRDataCorruptionIndicators_[fed3offset+1][spg3offset+2]++;
	    mapHTRproblem(dcc_,spigot);
	    if (debug_>0)std::cout <<"HTR Problem: Case 8"<<std::endl;
	    break;
	  case (15): //b15
	    HalfHTRDataCorruptionIndicators_[fed3offset+1][spg3offset+1]++;
	    mapHTRproblem(dcc_,spigot);
	    break;
	  case ( 7): //CK
	    HalfHTRDataCorruptionIndicators_[fed3offset+1][spg3offset+0]++;
	    mapHTRproblem(dcc_,spigot);
	    if (debug_>0)std::cout <<"HTR Problem: Case 7"<<std::endl;
	    break;
	  //\\case ( 5): //LW removed 2010.02.16
	  //\\  HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+2]++;
	  //\\  //Sometimes set spuriously at startup, per-fiber, .: Leniency: 8
	  //\\  if (HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+2] > 8) { 
	  //\\    if (debug_>0)std::cout <<"HTR Problem: Case 5"<<std::endl;
	  //\\    break; 
	  //\\  }
	  case ( 3): //L1 (previous L1A violated trigger rules)
	    DataFlowInd_[fed2offset+1][spg3offset+0]++; break;
	  case ( 1): //BZ
	    DataFlowInd_[fed2offset+0][spg3offset+1]++; break;
	  case ( 0): //OW
	    DataFlowInd_[fed2offset+0][spg3offset+2]++;
	  default: break;
	  }
	  meStatusWdCrate_->Fill(cratenum,i);
	  if      (cratenum == 0) meCrate0HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 1) meCrate1HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 2) meCrate2HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 3) meCrate3HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 4) meCrate4HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 5) meCrate5HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 6) meCrate6HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 7) meCrate7HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum == 9) meCrate9HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==10)meCrate10HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==11)meCrate11HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==12)meCrate12HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==13)meCrate13HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==14)meCrate14HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==15)meCrate15HTRStatus_ -> Fill(slotnum,i);
	  else if (cratenum ==17)meCrate17HTRStatus_ -> Fill(slotnum,i);
	} 
      }
    }
 
    // Fish out Front-End Errors from the precision channels
    const short unsigned int* daq_first, *daq_last, *tp_first, *tp_last;
    const HcalQIESample* qie_begin, *qie_end, *qie_work;
    
    // get pointers
    htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);
    qie_begin=(HcalQIESample*)daq_first;
    qie_end=(HcalQIESample*)(daq_last+1); // one beyond last..
    
    //TESTME_HCALRAWDATA//if (dccid==715 && spigot==5 && tevt_%3==0)
    //TESTME_HCALRAWDATA//  Chann_DataIntegrityCheck_[dcc_][16][spg2offset]++;    

    /// branch point between 2006-2011 data format and 2012+ data format
    if (htr.getFormatVersion() < 6 ) { //HcalHTRData::FORMAT_VERSION_COMPACT_DATA is 6

      int lastcapid=-1;
      int samplecounter=-1;
      int htrchan=-1; // Valid: [1,24]
      int chn2offset=0; 
      int NTS = htr.getNDD(); //number time slices, in precision channels

      ChannSumm_DataIntegrityCheck_  [fed2offset-1][spg2offset+0]=-NTS;//For normalization by client - NB! negative!
      // Run over DAQ words for this spigot
      for (qie_work=qie_begin; qie_work!=qie_end; qie_work++) {
	if (qie_work->raw()==0xFFFF)  // filler word
	  continue;
	//Beginning a channel's samples?
	if (( 1 + ( 3* (qie_work->fiber()-1) ) + qie_work->fiberChan() )  != htrchan) { //new channel starting
	  // A fiber [1..8] carries three fiber channels, each is [0..2]. Make htrchan [1..24]
	  htrchan= (3* (qie_work->fiber()-1) ) + qie_work->fiberChan(); 
	  chn2offset = (htrchan*3)+1;
	  --ChannSumm_DataIntegrityCheck_  [fed2offset-1][spg2offset-1];//event tally -- NB! negative!
	  --Chann_DataIntegrityCheck_[dcc_][chn2offset-1][spg2offset-1];//event tally -- NB! negative!
	  if (samplecounter !=-1) { //Wrap up the previous channel if there is one
	    //Check the previous digi for number of timeslices
	    if (((samplecounter != NTS) &&
		 (samplecounter != 1)             )
		//||
		//((htrchan==5) && (spigot==5) && (dcc_==5))
		)
	      { //Wrong DigiSize
		++ChannSumm_DataIntegrityCheck_  [fed2offset+0][spg2offset+0];
		++Chann_DataIntegrityCheck_[dcc_][chn2offset+0][spg2offset+0];
		mapChannproblem(dcc_,spigot,htrchan);
		if (debug_)std::cout <<"mapChannelProblem:  Wrong Digi Size"<<std::endl;
	      } 
	  } 	
	  //set up for this new channel
	  lastcapid=qie_work->capid();
	  samplecounter=1;} // fi (qie_work->fiberAndChan() != lastfibchan)
	else { //precision samples not the first timeslice
	  int hope = lastcapid +1;// What capid would we hope for here?
	  if (hope==4) hope = 0;  // What capid would we hope for here?
	  if (qie_work->capid() != hope){
	    ++ChannSumm_DataIntegrityCheck_  [fed2offset+1][spg2offset+0];
	    ++Chann_DataIntegrityCheck_[dcc_][chn2offset+1][spg2offset+0];
	    mapChannproblem(dcc_,spigot,htrchan);
	    if (debug_)std::cout <<"mapChannelProblem:  Wrong Cap ID"<<std::endl;
	  }
	  lastcapid=qie_work->capid();
	  samplecounter++;}
	//For every sample, whether the first of the channel or not, !DV, Er
	if (!(qie_work->dv())){
	  ++ChannSumm_DataIntegrityCheck_  [fed2offset+0][spg2offset+1];
	  ++Chann_DataIntegrityCheck_[dcc_][chn2offset+0][spg2offset+1];
	}
	if (qie_work->er()) {      // FEE - Front End Error
	  ++ChannSumm_DataIntegrityCheck_  [fed2offset+1][spg2offset+1];
	  ++Chann_DataIntegrityCheck_[dcc_][chn2offset+1][spg2offset+1]; 
	  mapChannproblem(dcc_,spigot,htrchan);
	  if (debug_)std::cout <<"mapChannelProblem:  FE Error"<<std::endl;	
	}
      } // for (qie_work = qie_begin;...)  end loop over all timesamples in this spigot
      //Wrap up the last channel
      //Check the last digi for number of timeslices
      if ((samplecounter != NTS) &&
	  (samplecounter != 1)            &&
	  (samplecounter !=-1)             ) { //Wrong DigiSize (unexpected num. timesamples)
	++ChannSumm_DataIntegrityCheck_  [fed2offset+0][spg2offset+0];
	++Chann_DataIntegrityCheck_[dcc_][chn2offset+0][spg2offset+0];
	mapChannproblem(dcc_,spigot,htrchan);
	if (debug_)std::cout <<"mapChannelProblem:  Wrong Digi Size (last digi)"<<std::endl;
      }
    } else { // this is the branch for unpacking the compact data format with per-channel headers
      const unsigned short* ptr_header=daq_first;
      const unsigned short* ptr_end=daq_last+1;
      int flavor, error_flags, capid0, channelid;
      // int samplecounter=-1;  // for a digisize check
      int htrchan=-1; // Valid: [1,24]
      int chn2offset=0; 
      int NTS = htr.getNDD(); //number time slices, in precision channels
      int Nchan = 3; // 3 channels per fiber
      while (ptr_header!=ptr_end) {
    	if (*ptr_header==0xFFFF) { // impossible filler word
    	  ptr_header++;
    	  continue;
    	}
	error_flags = capid0 = channelid = 0;
	// unpack the header word
	bool isheader=HcalHTRData::unpack_per_channel_header(*ptr_header,flavor,error_flags,capid0,channelid);
	if (!isheader) {
	  ptr_header++;
	  continue;
	}
	// A fiber [1..8] carries three fiber channels, each is [0..2]. Make htrchan [1..24]
	int fiber = 1 + ((channelid & 0x1C) >> 2); //Mask and shift to get bits [2:4]
	int chan = channelid & 0x3; //bits [0:1]
	htrchan = ((fiber -1) * Nchan) + chan + 1;  //ta-dah! I really wish everything counted from zero...
	chn2offset = ((htrchan-1)*3)+1; //For placing the errors on the histogram. Also very tidy. Sorry.
	//debug// if (error_flags) std::cout<<fiber<<","<<chan<<" = "<<htrchan<<"   @"<<chn2offset<<std::endl;
	ChannSumm_DataIntegrityCheck_  [fed2offset-1][spg2offset-1] -= NTS;//event tally -- NB: negative!
	Chann_DataIntegrityCheck_[dcc_][chn2offset-1][spg2offset-1] -= NTS;//event tally -- NB: negative!
	if (error_flags & 2) { //a CapId violation (non correct rotation)
	  ++ChannSumm_DataIntegrityCheck_  [fed2offset+1][spg2offset+0];
	  ++Chann_DataIntegrityCheck_[dcc_][chn2offset+1][spg2offset+0];
	  mapChannproblem(dcc_,spigot,htrchan);
	}
	if (error_flags & 1) { //an asserted Link-Error (Er)
	  ++ChannSumm_DataIntegrityCheck_  [fed2offset+1][spg2offset+1];
	  ++Chann_DataIntegrityCheck_[dcc_][chn2offset+1][spg2offset+1]; 
	  mapChannproblem(dcc_,spigot,htrchan);
	}

	for (ptr_header++;
	     ptr_header!=ptr_end && !HcalHTRData::is_channel_header(*ptr_header);
	     ptr_header++);      
      }
    }
    unsigned int fib1BCN = htr.getFib1OrbMsgBCN();
    unsigned int fib2BCN = htr.getFib2OrbMsgBCN();
    unsigned int fib3BCN = htr.getFib3OrbMsgBCN();
    unsigned int fib4BCN = htr.getFib4OrbMsgBCN();
    unsigned int fib5BCN = htr.getFib5OrbMsgBCN();
    unsigned int fib6BCN = htr.getFib6OrbMsgBCN();
    unsigned int fib7BCN = htr.getFib7OrbMsgBCN();
    unsigned int fib8BCN = htr.getFib8OrbMsgBCN();
    meFibBCN_->Fill(fib1BCN);
    meFibBCN_->Fill(fib2BCN);
    meFibBCN_->Fill(fib3BCN);
    meFibBCN_->Fill(fib4BCN);
    meFibBCN_->Fill(fib5BCN);
    meFibBCN_->Fill(fib6BCN);
    meFibBCN_->Fill(fib7BCN);
    meFibBCN_->Fill(fib8BCN);
    // Disable for now?
    meFib1OrbMsgBCN_->Fill(slotnum, cratenum, fib1BCN);
    meFib2OrbMsgBCN_->Fill(slotnum, cratenum, fib2BCN);
    meFib3OrbMsgBCN_->Fill(slotnum, cratenum, fib3BCN);
    meFib4OrbMsgBCN_->Fill(slotnum, cratenum, fib4BCN);
    meFib5OrbMsgBCN_->Fill(slotnum, cratenum, fib5BCN);
    meFib6OrbMsgBCN_->Fill(slotnum, cratenum, fib6BCN);
    meFib7OrbMsgBCN_->Fill(slotnum, cratenum, fib7BCN);
    meFib8OrbMsgBCN_->Fill(slotnum, cratenum, fib8BCN);
    
  } //  loop over spigots 
  return;
} // loop over DCCs void HcalRawDataMonitor::unpack(


// End LumiBlock
void HcalRawDataMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
					    const edm::EventSetup& c){
  
  
  ProblemsVsLB_HB->Fill(lumiSeg.luminosityBlock(),NumBadHB);
  ProblemsVsLB_HE->Fill(lumiSeg.luminosityBlock(),NumBadHE);
  ProblemsVsLB_HO->Fill(lumiSeg.luminosityBlock(),NumBadHO);
  ProblemsVsLB_HF->Fill(lumiSeg.luminosityBlock(),NumBadHF);
  ProblemsVsLB_HBHEHF->Fill(lumiSeg.luminosityBlock(),NumBadHB+NumBadHE+NumBadHF);
  ProblemsVsLB->Fill(lumiSeg.luminosityBlock(),NumBadHB+NumBadHE+NumBadHO+NumBadHF);
  
  // Reset current LS histogram, if it exists
  if (ProblemsCurrentLB)
    ProblemsCurrentLB->Reset();
  if (ProblemsCurrentLB)
    {
     
      ProblemsCurrentLB->setBinContent(0,0, levt_);  // underflow bin contains number of events
      ProblemsCurrentLB->setBinContent(1,1, NumBadHB*levt_);
      ProblemsCurrentLB->setBinContent(2,1, NumBadHE*levt_);
      ProblemsCurrentLB->setBinContent(3,1, NumBadHO*levt_);
      ProblemsCurrentLB->setBinContent(4,1, NumBadHF*levt_);
      ProblemsCurrentLB->setBinContent(5,1, NumBadHO0*levt_);
      ProblemsCurrentLB->setBinContent(6,1, NumBadHO12*levt_);
      ProblemsCurrentLB->setBinContent(7,1, NumBadHFLUMI*levt_);

    }

  for (int d=0; d<DEPTHBINS; d++) {
    for (int eta=0; eta<ETABINS; eta++) {
      for (int phi=0; phi<PHIBINS; phi++){
	uniqcounter[eta][phi][d] = 0.0;
      }
    }
  }

  UpdateMEs();
}
// EndRun -- Anything to do here?
void HcalRawDataMonitor::endRun(const edm::Run& run, const edm::EventSetup& c){}

// EndJob
void HcalRawDataMonitor::endJob(void){
  if (debug_>0) std::cout <<"HcalRawDataMonitor::endJob()"<<std::endl;
  if (enableCleanup_) cleanup(); // when do we force cleanup?
}

//No size checking; better have enough y-axis bins!
void HcalRawDataMonitor::label_ySpigots(MonitorElement* me_ptr, int ybins) {
  char label[32];
  for (int spig=0; spig<HcalDCCHeader::SPIGOT_COUNT; spig++) {
    snprintf(label, 32, "Spgt %02d", spig);
    me_ptr->setBinLabel((2+(spig*ybins)), label, 2); //margin of 1 at low value
  }
}

//No size checking; better have enough x-axis bins!
void HcalRawDataMonitor::label_xChanns(MonitorElement* me_ptr, int xbins) {
  char label[32];
  for (int ch=0; ch<HcalHTRData::CHANNELS_PER_SPIGOT; ch++) {
    snprintf(label, 32, "Ch %02d", ch+1);
    me_ptr->setBinLabel((2+(ch*xbins)), label, 1); //margin of 3 at low value
  }
}

//No size checking; better have enough x-axis bins!
void HcalRawDataMonitor::label_xFEDs(MonitorElement* me_ptr, int xbins) {
  char label[32];
  for (int thfed=0; thfed<NUMDCCS; thfed++) {
    snprintf(label, 32, "%03d", thfed+700);
    me_ptr->setBinLabel((2+(thfed*xbins)), label, 1); //margin of 1 at low value
  }
}

void HcalRawDataMonitor::labelHTRBits(MonitorElement* mePlot,unsigned int axisType) {

  if (axisType !=1 && axisType != 2) return;

  mePlot -> setBinLabel(1,"Overflow Warn",axisType);
  mePlot -> setBinLabel(2,"Buffer Busy",axisType);
  mePlot -> setBinLabel(3,"Empty Event",axisType);
  mePlot -> setBinLabel(4,"Rejected L1A",axisType);
  mePlot -> setBinLabel(5,"Invalid Stream",axisType);
  mePlot -> setBinLabel(6,"Latency Warn",axisType);
  mePlot -> setBinLabel(7,"OptDat Err",axisType);
  mePlot -> setBinLabel(8,"Clock Err",axisType);
  mePlot -> setBinLabel(9,"Bunch Err",axisType);
  mePlot -> setBinLabel(10,"b9",axisType);
  mePlot -> setBinLabel(11,"b10",axisType);
  mePlot -> setBinLabel(12,"b11",axisType);
  mePlot -> setBinLabel(13,"Test Mode",axisType);
  mePlot -> setBinLabel(14,"Histo Mode",axisType);
  mePlot -> setBinLabel(15,"Calib Trig",axisType);
  mePlot -> setBinLabel(16,"Bit15 Err",axisType);

  return;
}

void HcalRawDataMonitor::stashHDI(int thehash, HcalDetId thehcaldetid) {
  //Let's not allow indexing off the array...
  if ((thehash<0)||(thehash>(NUMDCCS*NUMSPIGS*HTRCHANMAX)))return;
  //...but still do the job requested.
  hashedHcalDetId_[thehash] = thehcaldetid;
}

//Debugging output for single half-HTRs (single spigot) 
//-->Class member debug_ usually passed for prtlvl argument.<--
void HcalRawDataMonitor::HTRPrint(const HcalHTRData& htr,int prtlvl){
  if (prtlvl == 1){ 
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    printf("Crate,Slot,ErrWord,Evt#,BCN:  %3i %4.1f %6X %7i %4X \n", cratenum,slotnum,htr.getErrorsWord(),htr.getL1ANumber(),htr.getBunchNumber());
    //    printf(" DLLunlk,TTCrdy:%2i %2i \n",htr.getDLLunlock(),htr.getTTCready());
  }
  // This one needs new version of HcalHTRData.h to activate
  else if (prtlvl == 2){
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    printf("Crate, Slot:%3i %4.1f \n", cratenum,slotnum);
    //    printf("  Ext Hdr: %4X %4X %4X %4X %4X %4X %4X %4X \n",htr.getExtHdr1(),htr.getExtHdr2(),htr.getExtHdr3(),htr.getExtHdr4(),htr.getExtHdr5(),htr.getExtHdr6(),htr.getExtHdr7(),htr.getExtHdr8());
  }

  else if (prtlvl == 3){
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    printf("Crate, Slot:%3i %4.1f", cratenum,slotnum);
    printf(" FibOrbMsgBCNs: %4X %4X %4X %4X %4X %4X %4X %4X \n",htr.getFib1OrbMsgBCN(),htr.getFib2OrbMsgBCN(),htr.getFib3OrbMsgBCN(),htr.getFib4OrbMsgBCN(),htr.getFib5OrbMsgBCN(),htr.getFib6OrbMsgBCN(),htr.getFib7OrbMsgBCN(),htr.getFib8OrbMsgBCN());
  }

  return;
}


void HcalRawDataMonitor::UpdateMEs (void ) {
  tevt_=0;
  if (meTevtHist_) tevt_= (int)meTevtHist_->getBinContent(1);
  NumBadHB=0;
  NumBadHE=0;
  NumBadHO=0;
  NumBadHF=0;
  NumBadHFLUMI=0;
  NumBadHO0=0;
  NumBadHO12=0;


  meLRBDataCorruptionIndicators_->setBinContent(0,0,tevt_);
  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      if (LRBDataCorruptionIndicators_  [x][y]) 
	meLRBDataCorruptionIndicators_->setBinContent(x+1,y+1,LRBDataCorruptionIndicators_[x][y]);
	
  meHalfHTRDataCorruptionIndicators_->setBinContent(0,0,tevt_);
  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      if (HalfHTRDataCorruptionIndicators_  [x][y])
	meHalfHTRDataCorruptionIndicators_->setBinContent(x+1,y+1,HalfHTRDataCorruptionIndicators_[x][y]);

  meChannSumm_DataIntegrityCheck_->setBinContent(0,0,tevt_);  	 
  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<TWO__SPGT; y++)
      if (ChannSumm_DataIntegrityCheck_[x][y]) 
	meChannSumm_DataIntegrityCheck_->setBinContent(x+1,y+1,ChannSumm_DataIntegrityCheck_[x][y]);

  for (int f=0; f<NUMDCCS; f++){
    meChann_DataIntegrityCheck_[f]->setBinContent(0,0,tevt_);
    for (int x=0; x<TWO_CHANN; x++)
      for (int y=0; y<TWO__SPGT; y++)      
	if (Chann_DataIntegrityCheck_[f][x][y])
	  meChann_DataIntegrityCheck_[f]->setBinContent(x+1,y+1,Chann_DataIntegrityCheck_ [f][x][y]);
  }
  
  meDataFlowInd_->setBinContent(0,0,tevt_);
  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<THREE_SPG; y++)      
      if (DataFlowInd_[x][y])
	meDataFlowInd_->setBinContent(x+1,y+1,DataFlowInd_[x][y]);
} //UpdateMEs

//Increment the NumBad counter for this LS, for this Hcal subdet
void HcalRawDataMonitor::whosebad(int subdet) {
//  if (subdet==HcalBarrel)       ++NumBadHB;
//  else if (subdet==HcalEndcap)  ++NumBadHE;
//  else if (subdet==HcalOuter)  
//    {
//      ++NumBadHO;
//      if (abs(ieta)<5) ++NumBadHO0;
//      else ++NumBadHO12;
//    }
//  else if (subdet==HcalForward)
//    {
//      ++NumBadHF;
//      if (depth==1 && (abs(ieta)==33 || abs(ieta)==34))
//	++NumBadHFLUMI;
//      else if (depth==2 && (abs(ieta)==35 || abs(ieta)==36))
//	++NumBadHFLUMI;
//    }
}

void HcalRawDataMonitor::mapDCCproblem(int dcc) {
  int myeta   = 0;
  int myphi   =-1;
  int mydepth = 0;
  HcalDetId HDI;
  //Light up all affected cells.
  for (int i=hashup(dcc); 
       i<hashup(dcc)+(NUMSPIGS*HTRCHANMAX); 
       i++) {
    HDI = hashedHcalDetId_[i];
    if (HDI==HcalDetId::Undefined) 
      continue;
    mydepth = HDI.depth();
    myphi   = HDI.iphi();
    myeta = CalcEtaBin(HDI.subdet(),
		       HDI.ieta(),
		       mydepth);
    //Protect against indexing off array
    if (myeta>=0 && myeta<85 &&
	(myphi-1)>=0 && (myphi-1)<72 &&
	(mydepth-1)>=0 && (mydepth-1)<4){
      problemfound[myeta][myphi-1][mydepth-1] = true;

      //exlcude the decommissioned HO ring2, except SiPMs 
      if(mydepth==4 && excludeHORing2_==true)
	if (abs(HDI.ieta())>=11 && abs(HDI.ieta())<=15  && !isSiPM(HDI.ieta(),HDI.iphi(),mydepth))
	  problemfound[myeta][myphi-1][mydepth-1] = false;
      
      if (debug_>0)
	std::cout<<" mapDCCproblem found error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<std::endl;
    }
  }
}
void HcalRawDataMonitor::mapHTRproblem(int dcc, int spigot) {
  int myeta = 0;
  int myphi   =-1;
  int mydepth = 0;
  HcalDetId HDI;
  //Light up all affected cells.
  for (int i=hashup(dcc,spigot); 
       i<hashup(dcc,spigot)+(HTRCHANMAX); //nice, linear hash....
       i++) {
    HDI = hashedHcalDetId_[i];
    if (HDI==HcalDetId::Undefined) {
      continue;
    }
    mydepth = HDI.depth();
    myphi   = HDI.iphi();
    myeta = CalcEtaBin(HDI.subdet(),
		       HDI.ieta(),
		       mydepth);
    //Protect against indexing off array
    if (myeta>=0 && myeta<85 &&
	(myphi-1)>=0 && (myphi-1)<72 &&
	(mydepth-1)>=0 && (mydepth-1)<4){
      problemfound[myeta][myphi-1][mydepth-1] = true;
      
      //exlcude the decommissioned HO ring2, except SiPMs 
      if(mydepth==4 && excludeHORing2_==true)
	if (abs(HDI.ieta())>=11 && abs(HDI.ieta())<=15  && !isSiPM(HDI.ieta(),HDI.iphi(),mydepth))
	  problemfound[myeta][myphi-1][mydepth-1] = false;
      
      if (debug_>0)
	std::cout<<" mapDCCproblem found error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<std::endl;
    }

  }
}   // void HcalRawDataMonitor::mapHTRproblem(...)

void HcalRawDataMonitor::mapChannproblem(int dcc, int spigot, int htrchan) {
  int myeta = 0;
  int myphi   =-1;
  int mydepth = 0;
  HcalDetId HDI;
  //Light up the affected cell.
  int i=hashup(dcc,spigot,htrchan); 
  HDI = HashToHDI(i);
  if (HDI==HcalDetId::Undefined) {
    return; // Do nothing at all, instead.
  } 
  mydepth = HDI.depth();
  myphi   = HDI.iphi();
  myeta = CalcEtaBin(HDI.subdet(),
		     HDI.ieta(),
		     mydepth);
  //Protect against indexing off array
  if (myeta>=0 && myeta<85 &&
      (myphi-1)>=0 && (myphi-1)<72 &&
      (mydepth-1)>=0 && (mydepth-1)<4){
    problemfound[myeta][myphi-1][mydepth-1] = true;

    //exlcude the decommissioned HO ring2, except SiPMs 
    if(mydepth==4 && excludeHORing2_==true)
      if (abs(HDI.ieta())>=11 && abs(HDI.ieta())<=15  && !isSiPM(HDI.ieta(),HDI.iphi(),mydepth))
	  problemfound[myeta][myphi-1][mydepth-1] = false;

    if (debug_>0)
      std::cout<<" mapDCCproblem found error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<std::endl;
  }
}   // void HcalRawDataMonitor::mapChannproblem(...)


DEFINE_FWK_MODULE(HcalRawDataMonitor);
