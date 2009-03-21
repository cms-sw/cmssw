#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDataFormatMonitor::HcalDataFormatMonitor() {
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

  for (int f=0; f<NUMDCCS; f++) {
    for (int s=0; s<15; s++) {
      UScount[f][s]=0;}}

  for (int x=0; x<RCDIX; x++)
    for (int y=0; y<RCDIY; y++)
      DCC_DataIntegrityCheck_      [x][y]=0;	

  for (int x=0; x<HHDIX; x++)
    for (int y=0; y<HHDIY; y++)
      HalfHTR_DataIntegrityCheck_  [x][y]=0;
  	 
  for (int x=0; x<CSDIX; x++)
    for (int y=0; y<HHDIY; y++)
      ChannSumm_DataIntegrityCheck_[x][y]=0;

  for (int f=0; f<NUMDCCS; f++)
    for (int x=0; x<CIX; x++)
      for (int y=0; y<CIY; y++)      
	Chann_DataIntegrityCheck_  [f][x][y]=0;

  for (unsigned int eta =0 ; eta < ETABINS; eta++) {
    for (unsigned int phi =0 ; phi < PHIBINS; phi++) {
      for (unsigned int depth =0 ; depth < DEPTHBINS; depth++) {
	problemcount[eta][phi][depth] = 0;   
	problemfound[eta][phi][depth] = false;   
      }
    }
  }

} // HcalDataFormatMonitor::HcalDataFormatMonitor()

HcalDataFormatMonitor::~HcalDataFormatMonitor() {}

void HcalDataFormatMonitor::reset(){}

void HcalDataFormatMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
  }
}


void HcalDataFormatMonitor::setup(const edm::ParameterSet& ps,
				  DQMStore* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  ievt_=0;
  baseFolder_ = rootFolder_+"DataFormatMonitor";

  if(fVerbosity) cout << "About to pushback fedUnpackList_" << endl;
  // Use this in CMSSW_2_2:
  firstFED_ = FEDNumbering::getHcalFEDIds().first; 	  
  for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++)
    // Use this in CMSSW_3_0 and above:
    //firstFED_ = FEDNumbering::MINHCALFEDID;
    //for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++) 
    {
      if(fVerbosity) cout << "[DFMon:]Pushback for fedUnpackList_: " << i <<endl;
      fedUnpackList_.push_back(i);
    }

  prtlvl_ = ps.getUntrackedParameter<int>("dfPrtLvl");
  dfmon_checkNevents = ps.getUntrackedParameter<int>("DataFormatMonitor_checkNevents",checkNevents_);

  if ( m_dbe ) {
    char* type;
    m_dbe->setCurrentFolder(baseFolder_);

    HWProblems_=m_dbe->book2D(" HardwareWatchCells",
			      " Hardware Watch Cells for HCAL",
			      etaBins_, etaMin_,etaMax_,
			      phiBins_,phiMin_,phiMax_);
    HWProblems_->setAxisTitle("i#eta",1);
    HWProblems_->setAxisTitle("i#phi",2);
    setupDepthHists2D(HWProblemsByDepth_," Hardware Watch Cells", "");
    
    meEVT_ = m_dbe->bookInt("Data Format Task Event Number");
    meEVT_->Fill(ievt_);    

    m_dbe->setCurrentFolder(baseFolder_ + "/HcalFEDChecking");
    
    type="FEDEntries";
    fedEntries_ = m_dbe->book1D(type,type,32,699.5,731.5);
    type="FEDFatal";
    fedFatal_ = m_dbe->book1D(type,type,32,699.5,731.5);

    m_dbe->setCurrentFolder(baseFolder_);
    type="Readout Chain DataIntegrity Check";
    meDCC_DataIntegrityCheck_ = m_dbe->book2D(type,type,
					      RCDIX,0,RCDIX,
					      RCDIY,0,RCDIY);
    meDCC_DataIntegrityCheck_->setBinLabel( 1," 0 702",1);
    meDCC_DataIntegrityCheck_->setBinLabel( 2," 0/703",1); //skip 3
    meDCC_DataIntegrityCheck_->setBinLabel( 4," 1 704",1);
    meDCC_DataIntegrityCheck_->setBinLabel( 5," 1/705",1); //skip 6
    meDCC_DataIntegrityCheck_->setBinLabel( 7," 2 718",1);
    meDCC_DataIntegrityCheck_->setBinLabel( 8," 2/719",1); //skip 9 
    meDCC_DataIntegrityCheck_->setBinLabel(10," 3 724",1);
    meDCC_DataIntegrityCheck_->setBinLabel(11," 3/725",1); //skip 12
    meDCC_DataIntegrityCheck_->setBinLabel(13," 4 700",1); 
    meDCC_DataIntegrityCheck_->setBinLabel(14," 4/701",1); //skip 15
    meDCC_DataIntegrityCheck_->setBinLabel(16," 5 706",1);
    meDCC_DataIntegrityCheck_->setBinLabel(17," 5/707",1); //skip 18
    meDCC_DataIntegrityCheck_->setBinLabel(19," 6 728",1);
    meDCC_DataIntegrityCheck_->setBinLabel(20," 6/729",1); //skip 21
    meDCC_DataIntegrityCheck_->setBinLabel(22," 7 726",1); 
    meDCC_DataIntegrityCheck_->setBinLabel(23," 7/727",1); //skip 24-27
    meDCC_DataIntegrityCheck_->setBinLabel(28," 9 720",1);
    meDCC_DataIntegrityCheck_->setBinLabel(29," 9/721",1); //skip 30
    meDCC_DataIntegrityCheck_->setBinLabel(31,"10 716",1); 
    meDCC_DataIntegrityCheck_->setBinLabel(32,"10/717",1); //skip 33
    meDCC_DataIntegrityCheck_->setBinLabel(34,"11 708",1);
    meDCC_DataIntegrityCheck_->setBinLabel(35,"11/709",1); //skip 36
    meDCC_DataIntegrityCheck_->setBinLabel(37,"12 722",1);
    meDCC_DataIntegrityCheck_->setBinLabel(38,"12/723",1); //skip 39
    meDCC_DataIntegrityCheck_->setBinLabel(40,"13 730",1);
    meDCC_DataIntegrityCheck_->setBinLabel(41,"13/731",1); //skip 42
    meDCC_DataIntegrityCheck_->setBinLabel(43,"14 714",1);
    meDCC_DataIntegrityCheck_->setBinLabel(44,"14/715",1); //skip 45
    meDCC_DataIntegrityCheck_->setBinLabel(46,"15 710",1);
    meDCC_DataIntegrityCheck_->setBinLabel(47,"15/711",1); //skip 48-51
    meDCC_DataIntegrityCheck_->setBinLabel(52,"17 712",1);
    meDCC_DataIntegrityCheck_->setBinLabel(53,"17/713",1); //leave 54
    meDCC_DataIntegrityCheck_->setBinLabel(22,"    TTS",2); // TTS state 	 
    meDCC_DataIntegrityCheck_->setBinLabel(21,"    Rdy",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(20,"    OFW",2); 		  
    meDCC_DataIntegrityCheck_->setBinLabel(19,"    BSY",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(18,"    SYN",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(17,"HTRStat",2); // HTR State  
    meDCC_DataIntegrityCheck_->setBinLabel(16,"     OW",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(15,"     BZ",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(14,"Num Mis",2); //Num Mis	 
    meDCC_DataIntegrityCheck_->setBinLabel(13,"    EvNt",2) 		 ;
    meDCC_DataIntegrityCheck_->setBinLabel(12,"    BcN",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(11,"    OrN",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel(10,"DataLos",2); //Data Lost  
    meDCC_DataIntegrityCheck_->setBinLabel( 9,"    HTR",2); 		  
    meDCC_DataIntegrityCheck_->setBinLabel( 8,"    LRB",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel( 7,"    DCC",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel( 6,"LinkErr",2); //LinkErrs	 
    meDCC_DataIntegrityCheck_->setBinLabel( 5,"    FEE",2); 		  
    meDCC_DataIntegrityCheck_->setBinLabel( 4,"    HTR",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel( 3,"FmtErrs",2); 		 
    meDCC_DataIntegrityCheck_->setBinLabel( 2,"   Size",2); //FmtErrs    
    meDCC_DataIntegrityCheck_->setBinLabel( 1,"       ",2);

    m_dbe->setCurrentFolder(baseFolder_ + "/HTR Plots");
    type="Half-HTR DataIntegrity Check";
    meHalfHTR_DataIntegrityCheck_= m_dbe->book2D(type,type,
						 HHDIX,0,HHDIX,
						 HHDIY,0,HHDIY);
    meHalfHTR_DataIntegrityCheck_->setBinLabel( 2,"700",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel( 5,"701",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel( 8,"702",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(11,"703",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(14,"704",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(17,"705",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(20,"706",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(23,"707",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(26,"708",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(29,"709",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(32,"710",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(35,"711",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(38,"712",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(41,"713",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(44,"714",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(47,"715",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(50,"716",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(53,"717",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(56,"718",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(59,"719",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(62,"720",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(65,"721",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(68,"722",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(71,"723",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(74,"724",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(77,"725",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(80,"726",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(83,"727",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(86,"728",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(89,"729",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(92,"730",1);
    meHalfHTR_DataIntegrityCheck_->setBinLabel(95,"731",1);
    label_ySpigots(meHalfHTR_DataIntegrityCheck_, 4); // 3 bins + 1 margin each spgt

    type = "Channel Integrity Summarized by Spigot";
    meChannSumm_DataIntegrityCheck_= m_dbe->book2D(type,type,
						   CSDIX,0,CSDIX,
						   HHDIY,0,HHDIY);
    label_xFEDs (meChannSumm_DataIntegrityCheck_, 3); // 2 bins + 1 margin per ch.
    label_ySpigots(meChannSumm_DataIntegrityCheck_, 4); // 3 bins + 1 margin per spgt
 
    m_dbe->setCurrentFolder(baseFolder_ + "/HTR Plots/ Channel Data Integrity");
    char label[10];
    for (int f=0; f<NUMDCCS; f++){      
      sprintf(label, "FED %03d Channel Integrity", f+700);
      meChann_DataIntegrityCheck_[f] =  m_dbe->book2D(label,label,
						      CIX,0,CIX,
						      CIY,0,CIY);
      label_xChanns (meChann_DataIntegrityCheck_[f], 3); // 2 bins + 1 margin per ch.
      label_ySpigots(meChann_DataIntegrityCheck_[f], 3); // 2 bins + 1 margin per spgt
      ;}
      
    m_dbe->setCurrentFolder(baseFolder_ + "/DCC Plots");
    type = "BCN from DCCs";
    medccBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
    medccBCN_->setAxisTitle("BCN",1);
    medccBCN_->setAxisTitle("# of Entries",2);


    type = "DCC Error and Warning";
    meDCCErrorAndWarnConditions_ = m_dbe->book2D(type,type,32,699.5,731.5, 25,0.5,25.5);
    meDCCErrorAndWarnConditions_->setAxisTitle("HCAL FED ID", 1);      
    meDCCErrorAndWarnConditions_->setBinLabel( 1, "700", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 2, "701", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 3, "702", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 4, "703", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 5, "704", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 6, "705", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 7, "706", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 8, "707", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 9, "708", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(10, "709", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(11, "710", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(12, "711", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(13, "712", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(14, "713", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(15, "714", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(16, "715", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(17, "716", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(18, "717", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(19, "718", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(20, "719", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(21, "720", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(22, "721", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(23, "722", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(24, "723", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(25, "724", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(26, "725", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(27, "726", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(28, "727", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(29, "728", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(30, "729", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(31, "730", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(32, "731", 1);
    meDCCErrorAndWarnConditions_->setBinLabel( 1, "MisM S14", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 2, "MisM S13", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 3, "MisM S12", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 4, "MisM S11", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 5, "MisM S10", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 6, "MisM S9", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 7, "MisM S8", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 8, "MisM S7", 2);
    meDCCErrorAndWarnConditions_->setBinLabel( 9, "MisM S6", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(10, "MisM S5", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(11, "MisM S4", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(12, "MisM S3", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(13, "MisM S2", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(14, "MisM S1", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(15, "MisM S0(top)", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(16, "TTS_OFW", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(17, "TTS_BSY", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(18, "TTS_SYN", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(19, "L1A_EvN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(20, "L1A_BcN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(21, "CT_EvN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(22, "CT_BcN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(23, "OrbitLenEr", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(24, "TTC_SingEr", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(25, "TTC_DoubEr", 2);

    type = "DCC Ev Fragment Size Distribution";
    meFEDRawDataSizes_=m_dbe->book1D(type,type,12000,-0.5,12000.5);
    meFEDRawDataSizes_->setAxisTitle("# of bytes",1);
    meFEDRawDataSizes_->setAxisTitle("# of Event Fragments",2);

    type = "DCC Nonzero Spigot Conditions";
    meDCCSummariesOfHTRs_ = m_dbe->book2D(type,type,32,699.5,731.5, 20,0.5,20.5);
    meDCCSummariesOfHTRs_->setAxisTitle("HCAL FED ID", 1);
    meDCCSummariesOfHTRs_->setBinLabel(1, "HTR OFW", 2);
    meDCCSummariesOfHTRs_->setBinLabel(2, "HTR BSY", 2);
    meDCCSummariesOfHTRs_->setBinLabel(3, "Empty Events", 2);
    meDCCSummariesOfHTRs_->setBinLabel(4, "L1A Reject", 2);
    meDCCSummariesOfHTRs_->setBinLabel(5, "Latency Er", 2);
    meDCCSummariesOfHTRs_->setBinLabel(6, "Latncy Warn", 2);
    meDCCSummariesOfHTRs_->setBinLabel(7, "Optcl Data Err", 2);
    meDCCSummariesOfHTRs_->setBinLabel(8, "Clock", 2);
    meDCCSummariesOfHTRs_->setBinLabel(9, "CorrHamm LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(10, "UncorrHam", 2);
    meDCCSummariesOfHTRs_->setBinLabel(11, "LRB Block OvF", 2);
    meDCCSummariesOfHTRs_->setBinLabel(12, "LRB EvN Hdr/Tlr", 2);
    meDCCSummariesOfHTRs_->setBinLabel(13, "FIFOs Empty", 2);
    meDCCSummariesOfHTRs_->setBinLabel(14, "LRB Trunct", 2);
    meDCCSummariesOfHTRs_->setBinLabel(15, "LRB No Hdr/tlr", 2);
    meDCCSummariesOfHTRs_->setBinLabel(16, "Odd 16-Bit Wd Cnt", 2);
    meDCCSummariesOfHTRs_->setBinLabel(17, "Spgt E not P", 2);
    meDCCSummariesOfHTRs_->setBinLabel(18, "Spgt BcN Mis", 2);
    meDCCSummariesOfHTRs_->setBinLabel(19, "P not V", 2);
    meDCCSummariesOfHTRs_->setBinLabel(20, "Trunct by LRB", 2);

    // These status bits set in the header could be useful: They're set once, and stay on
    // even if the event in which they're set is not analyzed...

    ///type = "DCC Status Flags (Nonzero Error Counters)";
    ///meDCCStatusFlags_ = m_dbe->book2D(type,type,32,699.5,731.5,10,0.5,10.5);
    ///meDCCStatusFlags_->setAxisTitle("HCAL FED ID", 1);      
    ///meDCCStatusFlags_->setBinLabel(1, "Saw OFW", 2);
    ///meDCCStatusFlags_->setBinLabel(2, "Saw BSY", 2);
    ///meDCCStatusFlags_->setBinLabel(3, "Saw SYN", 2);
    ///meDCCStatusFlags_->setBinLabel(4, "MxMx_L1AEvN", 2);
    ///meDCCStatusFlags_->setBinLabel(5, "MxMx_L1ABcN", 2);
    ///meDCCStatusFlags_->setBinLabel(6, "MxMx_CT-EvN", 2);
    ///meDCCStatusFlags_->setBinLabel(7, "MxMx_CT-BCN", 2);
    ///meDCCStatusFlags_->setBinLabel(8, "BC0 Spacing", 2);
    ///meDCCStatusFlags_->setBinLabel(9, "TTCSingErr", 2);
    ///meDCCStatusFlags_->setBinLabel(10, "TTCDoubErr", 2);

    type = "Event Fragment Size for each FED";
    meEvFragSize_ = m_dbe->bookProfile(type,type,32,699.5,731.5,100,-1000.0,12000.0,"");
    type = "All Evt Frag Sizes";
    meEvFragSize2_ =  m_dbe->book2D(type,type,64,699.5,731.5, 2000,0,12000);

    type = "Num Event Frags by FED";
    meFEDId_=m_dbe->book1D(type, type, 32, 699.5, 731.5);
    meFEDId_->setAxisTitle("HCAL FED ID",1);

    type = "Spigot Format Errors";
    meSpigotFormatErrors_=  m_dbe->book1D(type,type,50,-0.5,49.5);
    meSpigotFormatErrors_->setAxisTitle("# of Errors",1);
    meSpigotFormatErrors_->setAxisTitle("# of Events",2);

    m_dbe->setCurrentFolder(baseFolder_ + "/DCC Plots/ZZ DCC Expert Plots");

    type = "Common Data Format violations";
    meCDFErrorFound_ = m_dbe->book2D(type,type,32,699.5,731.5,10,0.5,10.5);
    meCDFErrorFound_->setAxisTitle("HCAL FED ID", 1);
    meCDFErrorFound_->setBinLabel(1, "Hdr1BitUnset", 2);
    meCDFErrorFound_->setBinLabel(2, "FmtNumChange", 2);
    meCDFErrorFound_->setBinLabel(3, "EvTypChange", 2);
    meCDFErrorFound_->setBinLabel(4, "BOE not '0x5'", 2);
    meCDFErrorFound_->setBinLabel(5, "Hdr2Bit Set", 2);
    meCDFErrorFound_->setBinLabel(6, "Hdr1 36-59", 2);
    meCDFErrorFound_->setBinLabel(7, "BOE not 0", 2);
    meCDFErrorFound_->setBinLabel(8, "Trlr1Bit Set", 2);
    meCDFErrorFound_->setBinLabel(9, "Size Error", 2);
    meCDFErrorFound_->setBinLabel(10, "TrailerBad", 2);

    type = "DCC Event Format violation";
    meDCCEventFormatError_ = m_dbe->book2D(type,type,32,699.5,731.5,4,0.5,4.5);
    meDCCEventFormatError_->setAxisTitle("HCAL FED ID", 1);
    meDCCEventFormatError_->setBinLabel(1, "FmtVersChng", 2);
    meDCCEventFormatError_->setBinLabel(2, "StrayBits", 2);
    meDCCEventFormatError_->setBinLabel(3, "HTRStatusPad", 2);
    meDCCEventFormatError_->setBinLabel(4, "32bitPadErr", 2);
    //meDCCEventFormatError_->setBinLabel(5, "Spigot Error Flag Miscalculated", 2);      
    //meDCCEventFormatError_->setBinLabel(7, "LRB Truncation Bit MisCopied", 2);	       
    //meDCCEventFormatError_->setBinLabel(8, "32-Bit Padding Word Needed But Absent", 2);
    //meDCCEventFormatError_->setBinLabel(9, "Event Size Internally Misdescribed", 2);

    type = "FED Error Map from Unpacker Report";
    meFEDerrorMap_ = m_dbe->book1D(type,type,33,699.5,732.5);
    meFEDerrorMap_->setAxisTitle("Dcc Id",1);
    meFEDerrorMap_->setAxisTitle("# of Errors",2);

    m_dbe->setCurrentFolder(baseFolder_ + "/HTR Plots");

    type = "Fraction UnSuppressed Events";
    meUSFractSpigs_ = m_dbe->book1D(type,type,481,0,481);
    for(int f=0; f<NUMDCCS; f++) {
      sprintf(label, "FED 7%02d", f);
      meUSFractSpigs_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
      for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	sprintf(label, "sp%02d", s);
	meUSFractSpigs_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

    type = "BCN Difference Between Ref HTR and DCC";
    meBCNCheck_ = m_dbe->book1D(type,type,501,-250.5,250.5);
    meBCNCheck_->setAxisTitle("htr BCN - dcc BCN",1);

    type = "BCN Inconsistent - HTR vs Ref HTR";
    meBCNSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    meBCNSynch_->setAxisTitle("Slot #",1);
    meBCNSynch_->setAxisTitle("Crate #",2);

    type = "BCN from HTRs";
    meBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
    meBCN_->setAxisTitle("BCN",1);
    meBCN_->setAxisTitle("# of Entries",2);

    type = "EvN Difference Between Ref HTR and DCC";
    meEvtNCheck_ = m_dbe->book1D(type,type,601,-300.5,300.5);
    meEvtNCheck_->setAxisTitle("htr Evt # - dcc Evt #",1);

    type = "EvN Inconsistent - HTR vs Ref HTR";
    meEvtNumberSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    meEvtNumberSynch_->setAxisTitle("Slot #",1);
    meEvtNumberSynch_->setAxisTitle("Crate #",2);

    type = "HBHE Data Format Error Word";
    DCC_ErrWd_HBHE =  m_dbe->book1D(type,type,16,-0.5,15.5);
    labelHTRBits(DCC_ErrWd_HBHE,1);

    type = "HF Data Format Error Word";
    DCC_ErrWd_HF =  m_dbe->book1D(type,type,16,-0.5,15.5);
    labelHTRBits(DCC_ErrWd_HF,1);
  
    type = "HO Data Format Error Word";
    DCC_ErrWd_HO = m_dbe->book1D(type,type,16,-0.5,15.5);
    labelHTRBits(DCC_ErrWd_HO,1);

    int maxbits = 16;//Look at all 16 bits of the Error Words
    type = "HTR Error Word by Crate";
    meErrWdCrate_ = m_dbe->book2D(type,type,18,-0.5,17.5,maxbits,-0.5,maxbits-0.5);
    meErrWdCrate_ -> setAxisTitle("Crate #",1);
    labelHTRBits(meErrWdCrate_,2);

    type = "Unpacking - HcalHTRData check failures";
    meInvHTRData_= m_dbe->book2D(type,type,16,-0.5,15.5,32,699.5,731.5);
    meInvHTRData_->setAxisTitle("Spigot #",1);
    meInvHTRData_->setAxisTitle("DCC #",2);

    m_dbe->setCurrentFolder(baseFolder_ + "/HTR Plots/ZZ HTR Expert Plots");

    type = "BCN of Fiber Orbit Message";
    meFibBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
    meFibBCN_->setAxisTitle("BCN of Fib Orb Msg",1);


    type = "HTR Error Word - Crate 0";
    meCrate0HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate0HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate0HTRErr_,2);

    type = "HTR Error Word - Crate 1";
    meCrate1HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate1HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate1HTRErr_,2);

    type = "HTR Error Word - Crate 2";
    meCrate2HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate2HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate2HTRErr_,2);

    type = "HTR Error Word - Crate 3";
    meCrate3HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate3HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate3HTRErr_,2);

    type = "HTR Error Word - Crate 4";
    meCrate4HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate4HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate4HTRErr_,2);

    type = "HTR Error Word - Crate 5";
    meCrate5HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate5HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate5HTRErr_,2);

    type = "HTR Error Word - Crate 6";
    meCrate6HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate6HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate6HTRErr_,2);

    type = "HTR Error Word - Crate 7";
    meCrate7HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate7HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate7HTRErr_,2);

    type = "HTR Error Word - Crate 8";
    meCrate8HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate8HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate8HTRErr_,2);

    type = "HTR Error Word - Crate 9";
    meCrate9HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate9HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate9HTRErr_,2);

    type = "HTR Error Word - Crate 10";
    meCrate10HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate10HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate10HTRErr_,2);

    type = "HTR Error Word - Crate 11";
    meCrate11HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate11HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate11HTRErr_,2);

    type = "HTR Error Word - Crate 12";
    meCrate12HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate12HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate12HTRErr_,2);

    type = "HTR Error Word - Crate 13";
    meCrate13HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate13HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate13HTRErr_,2);

    type = "HTR Error Word - Crate 14";
    meCrate14HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate14HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate14HTRErr_,2);

    type = "HTR Error Word - Crate 15";
    meCrate15HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate15HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate15HTRErr_,2);

    type = "HTR Error Word - Crate 16";
    meCrate16HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate16HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate16HTRErr_,2);

    type = "HTR Error Word - Crate 17";
    meCrate17HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate17HTRErr_ ->setAxisTitle("Slot #",1);
    labelHTRBits(meCrate17HTRErr_,2);
    
    // Firmware version
    type = "HTR Firmware Version";
    //  Maybe change to Profile histo eventually
    //meFWVersion_ = m_dbe->bookProfile(type,type,18,-0.5,17.5,245,10.0,255.0,"");
    meFWVersion_ = m_dbe->book2D(type,type ,18,-0.5,17.5,180,75.5,255.5);
    meFWVersion_->setAxisTitle("Crate #",1);
    meFWVersion_->setAxisTitle("HTR Firmware Version",2);

    m_dbe->setCurrentFolder(baseFolder_ + "/ZZ HCal-Wide Expert Plots");

    type = "Num Bad Quality Digis -DV bit-Err bit-Cap Rotation";
    meBadQualityDigis_=  m_dbe->book1D(type,type,9100,-1,9099);
    meBadQualityDigis_->setAxisTitle("# of Bad Digis",1);
    meBadQualityDigis_->setAxisTitle("# of Events",2);

    type = "Num Unmapped Digis";
    meUnmappedDigis_=  m_dbe->book1D(type,type,9100,-1,9099);
    meUnmappedDigis_->setAxisTitle("# of Unmapped Digis",1);
    meUnmappedDigis_->setAxisTitle("# of Events",2);

    type = "Num Unmapped Trigger Primitive Digis";
    meUnmappedTPDigis_=  m_dbe->book1D(type,type,9100,-1,9099);
    meUnmappedTPDigis_->setAxisTitle("# of Unmapped Trigger Primitive Digis",1);
    meUnmappedTPDigis_->setAxisTitle("# of Events",2);

    /* Disable these histos for now
       type = "Fiber 1 Orbit Message BCN";
       meFib1OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 2 Orbit Message BCN";
       meFib2OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 3 Orbit Message BCN";
       meFib3OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 4 Orbit Message BCN";
       meFib4OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 5 Orbit Message BCN";
       meFib5OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 6 Orbit Message BCN";
       meFib6OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 7 Orbit Message BCN";
       meFib7OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       type = "Fiber 8 Orbit Message BCN";
       meFib8OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    */
  
  }

  return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw, 
					 const HcalUnpackerReport& report, 
					 const HcalElectronicsMap& emap){
  if(!m_dbe) { 
    printf("HcalDataFormatMonitor::processEvent DQMStore not instantiated!!!\n");  
    return;}
  
  ievt_++;
  meEVT_->Fill(ievt_);

  meSpigotFormatErrors_->Fill(report.spigotFormatErrors());
  meBadQualityDigis_->Fill(report.badQualityDigis());
  meUnmappedDigis_->Fill(report.unmappedDigis());
  meUnmappedTPDigis_->Fill(report.unmappedTPDigis());

  lastEvtN_ = -1;
  lastBCN_ = -1;

  // Loop over all FEDs reporting the event, unpacking if good.
  for (vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    if (fed.size()<12) continue;  //At least the size of headers and trailers of a DCC.
    unpack(fed,emap); //Interpret data, fill histograms, everything.
  }

  // Any problem worth mapping, anywhere?
  for (unsigned int eta =0 ; eta < ETABINS; eta++) {
    for (unsigned int phi =0 ; phi < PHIBINS; phi++) {
      for (unsigned int depth =0 ; depth < 4; depth++) {
	if (problemfound[eta][phi][depth]) 
	  ++problemcount[eta][phi][depth];   
      }
    }
  }
  if (0== (ievt_ % dfmon_checkNevents))
    UpdateMEs();
  //Transfer this event's problem info to 
  for (unsigned int eta =0 ; eta < ETABINS; eta++) {
    for (unsigned int phi =0 ; phi < PHIBINS; phi++) {
      for (unsigned int depth =0 ; depth < 4; depth++) {
	problemfound[eta][phi][depth] = false;   
      }
    }
  }

  for(unsigned int i=0; i<report.getFedsError().size(); i++){
    // Take the ith entry in the vector of FED IDs
    // with zero-size FEDRawData, size < 24, 
    const int m = report.getFedsError()[i];
    const FEDRawData& fed = rawraw.FEDData(m);
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
    if(!dccHeader) continue;
    int dccid=dccHeader->getSourceId();
    meFEDerrorMap_->Fill(dccid);}

  return;
} //void HcalDataFormatMonitor::processEvent()

// Process one FED's worth (one DCC's worth) of the event data.
void HcalDataFormatMonitor::unpack(const FEDRawData& raw, 
				   const HcalElectronicsMap& emap){
  // get the DCC header & trailer (or bail out)
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  // FED id declared in the metadata
  int dccid=dccHeader->getSourceId();
  //Force 0<= dcc_ <= 31
  int dcc_=max(0,dccid-700);  
  dcc_ = min(dcc_,31);       
  if(fVerbosity) cout << "DCC " << dccid << endl;
  //There should never be HCAL DCCs reporting a fed id outside [700:731]
  meFEDId_->Fill(dccid);

  //Before all else, how much data are we dealing with here?
  uint64_t* lastDataWord = (uint64_t*) ( raw.data()+raw.size()-(2*sizeof(uint64_t)) );
  int EvFragLength = ((*lastDataWord>>32)*8); // In bytes.
  EvFragLength = raw.size(); // Same source, at root, or a good check?
  meFEDRawDataSizes_->Fill(EvFragLength);
  meEvFragSize_ ->Fill(dccid, EvFragLength);
  meEvFragSize2_ ->Fill(dccid, EvFragLength);

  //DataIntegrity histogram bins
  int bin=0; 
  if (( (dcc_) % 2 )==0) //...lucky that odd FED ID's all have slot 19....
    bin = 3*( (int)DIMbin[dcc_]);
  else 
    bin = 1 + (3*(int)(DIMbin[dcc_]-0.5));
  int halfhtrDIM_x, halfhtrDIM_y;
  int chsummDIM_x, chsummDIM_y;
  int channDIM_x, channDIM_y;
  chsummDIM_x = halfhtrDIM_x = 1 + (dcc_)*3;
  
  //Orbit, BunchCount, and Event Numbers
  unsigned long dccEvtNum = dccHeader->getDCCEventNumber();
  int dccBCN = dccHeader->getBunchId();
  int dccOrN = dccHeader->getOrbitNumber();
  medccBCN_ -> Fill(dccBCN);

  ////////// Histogram problems with the Common Data Format compliance;////////////
  bool CDFProbThisDCC = false; 
  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()) {
    meCDFErrorFound_->Fill(dccid, 1);
    CDFProbThisDCC = true; 
  }
  /* 2 */ //Make sure a reference CDF Version value has been recorded for this dccid
  CDFvers_it = CDFversionNumber_list.find(dccid);
  if (CDFvers_it  == CDFversionNumber_list.end()) {
    CDFversionNumber_list.insert(pair<int,short>
				 (dccid,dccHeader->getCDFversionNumber() ) );
    CDFvers_it = CDFversionNumber_list.find(dccid);
  } // then check against it.
  if (dccHeader->getCDFversionNumber()!= CDFvers_it->second) {
    meCDFErrorFound_->Fill(dccid,2);
    CDFProbThisDCC = true; 
  }
  /* 3 */ //Make sure a reference CDF EventType value has been recorded for this dccid
  CDFEvT_it = CDFEventType_list.find(dccid);
  if (CDFEvT_it  == CDFEventType_list.end()) {
    CDFEventType_list.insert(pair<int,short>
			     (dccid,dccHeader->getCDFEventType() ) );
    CDFEvT_it = CDFEventType_list.find(dccid);
  } // then check against it.
  if (dccHeader->getCDFEventType()!= CDFEvT_it->second) {
    meCDFErrorFound_->Fill(dccid,3);
    CDFProbThisDCC = true; 
  }
  /* 4 */ //There should always be a '5' in CDF Header word 0, bits [63:60]
  if (dccHeader->BOEshouldBe5Always()!=5) {
    meCDFErrorFound_->Fill(dccid, 4);
    CDFProbThisDCC = true; 
  }
  /* 5 */ //There should never be a third CDF Header word indicated.
  if (dccHeader->thereIsAThirdCDFHeaderWord()) {
    meCDFErrorFound_->Fill(dccid, 5);
    CDFProbThisDCC = true; 
  }
  /* 6 */ //Make sure a reference value of Reserved Bits has been recorded for this dccid
  CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  if (CDFReservedBits_it  == CDFReservedBits_list.end()) {
    CDFReservedBits_list.insert(pair<int,short>
				(dccid,dccHeader->getSlink64ReservedBits() ) );
    CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  } // then check against it.
  if ((int) dccHeader->getSlink64ReservedBits()!= CDFReservedBits_it->second) {
    meCDFErrorFound_->Fill(dccid,6);
    CDFProbThisDCC = true; 
  }
  /* 7 */ //There should always be 0x0 in CDF Header word 1, bits [63:60]
  if (dccHeader->BOEshouldBeZeroAlways() !=0) {
    meCDFErrorFound_->Fill(dccid, 7);
    CDFProbThisDCC = true; 
  }
  /* 8 */ //There should only be one trailer
  if (trailer.moreTrailers()) {
    meCDFErrorFound_->Fill(dccid, 8);
    CDFProbThisDCC = true; 
  }
  //  if trailer.
  /* 9 */ //CDF Trailer [55:30] should be the # 64-bit words in the EvFragment
  if ((uint64_t) raw.size() != ( (uint64_t) trailer.lenght()*sizeof(uint64_t)) )  //The function name is a typo! Awesome.
    {
      meCDFErrorFound_->Fill(dccid, 9);
      CDFProbThisDCC = true; 
    }
  /*10 */ //There is a rudimentary sanity check built into the FEDTrailer class
  if (!trailer.check()) {
    meCDFErrorFound_->Fill(dccid, 10);
    CDFProbThisDCC = true; 
  }
  if (CDFProbThisDCC) {
    fillzoos(6,dccid);
    //Set the problem flag for the ieta, iphi of any channel in this DCC
    mapDCCproblem(dcc_);
  }
  if (CDFProbThisDCC)
    fedFatal_->Fill(dccid);
  fedEntries_->Fill(dccid);

  CDFProbThisDCC = false;  // reset for the next go-round.
  
  char CRC_err;
  for(int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    CRC_err = ((dccHeader->getSpigotSummary(i) >> 10) & 0x00000001);
    if (CRC_err) {
      fillzoos(5,dccid);
      //Set the problem flag for the ieta, iphi of any channel in this DCC
      mapHTRproblem(dcc_, i);  
    }
  }
  
  // Unlovely. There are N bytes in the DCC's raw data, the last at raw.data(N-1).
  // They're in words of 8 bytes.
  // Take the last word, take its low byte, and the high 4 bits are the TTS state. Easy.
  char TTS_state = ((raw.data()[raw.size()-8]>>4) & 0x0F);

  //  char TTS_state=(char) dccHeader->getAcceptTimeTTS();
  if (TTS_state & 0x8) /*RDY*/ 
    ++DCC_DataIntegrityCheck_[bin][20];
  if (TTS_state & 0x4) /*BSY*/ {
    ++DCC_DataIntegrityCheck_[bin][18];
    ///\\\///DATAFORMAT_PROBLEM_ZOO-> Fill(10);
  }
  if (TTS_state & 0x2) /*SYN*/ {
    ++DCC_DataIntegrityCheck_[bin][17];
    ++DCC_DataIntegrityCheck_[bin][ 6];}          // DCC lost data
  if (TTS_state & 0x1) /*OFW*/ {
    ++DCC_DataIntegrityCheck_[bin][19];
    ///\\\///DATAFORMAT_PROBLEM_ZOO-> Fill(9);
    mapDCCproblem(dcc_);}

  ////////// Histogram problems with DCC Event Format compliance;////////////
  /* 1 */ //Make sure a reference value of the DCC Event Format version has been noted for this dcc.
  DCCEvtFormat_it = DCCEvtFormat_list.find(dccid);
  if (DCCEvtFormat_it == DCCEvtFormat_list.end()) {
    DCCEvtFormat_list.insert(pair<int,short>
			     (dccid,dccHeader->getDCCDataFormatVersion() ) );
    DCCEvtFormat_it = DCCEvtFormat_list.find(dccid);
  } // then check against it.
  if (dccHeader->getDCCDataFormatVersion()!= DCCEvtFormat_it->second) 
    meDCCEventFormatError_->Fill(dccid,1);
  /* 2 */ //Check for ones where there should always be zeros
  if (false) //dccHeader->getByte1Zeroes() || dccHeader->getByte3Zeroes() || dccHeader->getByte567Zeroes()) 
    meDCCEventFormatError_->Fill(dccid,2);
  /* 3 */ //Check that there are zeros following the HTR Status words.
  int SpigotPad = HcalDCCHeader::SPIGOT_COUNT;
  if (  ((uint64_t) dccHeader->getSpigotSummary(SpigotPad)  ) 
	| ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+1)) 
	| ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+2))  != 0)
    meDCCEventFormatError_->Fill(dccid,3);
  /* 4 */ //Check that there are zeros following the HTR Payloads, if needed.
  int nHTR32BitWords=0;
  // add up all the declared HTR Payload lengths
  for(int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    nHTR32BitWords += dccHeader->getSpigotDataLength(i);  }
  // if it's an odd number, check for the padding zeroes
  if (( nHTR32BitWords % 2) == 1) {
    uint64_t* lastDataWord = (uint64_t*) ( raw.data()+raw.size()-(2*sizeof(uint64_t)) );
    if ((*lastDataWord>>32) != 0x00000000)
      meDCCEventFormatError_->Fill(dccid, 4);
  }
  
  ////////// Histogram Errors and Warnings from the DCC;////////////
  /* [1:15] */ //Histogram HTR Status Bits from the DCC Header
  for(int i=1; i<=HcalDCCHeader::SPIGOT_COUNT; i++)  
    // One bit: data missing || mismatch <EvN, BcN, || OrN>
    if (dccHeader->getSpigotErrorFlag(i))  meDCCErrorAndWarnConditions_->Fill(dccid, i);
  /* [16:25] */ //Histogram DCC Error and Warning Counters being nonzero
  if (TTS_state & 0x1)                  meDCCErrorAndWarnConditions_->Fill(dccid,16);
  if (TTS_state & 0x4)                  meDCCErrorAndWarnConditions_->Fill(dccid,17);
  if (TTS_state & 0x2)                  meDCCErrorAndWarnConditions_->Fill(dccid,18);
  if (dccHeader->SawL1A_EvN_MxMx()   )  meDCCErrorAndWarnConditions_->Fill(dccid,19);
  if (dccHeader->SawL1A_BcN_MxMx()   )  meDCCErrorAndWarnConditions_->Fill(dccid,20);
  if (dccHeader->SawCT_EvN_MxMx()    )  meDCCErrorAndWarnConditions_->Fill(dccid,21);
  if (dccHeader->SawCT_BcN_MxMx()    )  meDCCErrorAndWarnConditions_->Fill(dccid,22);
  if (dccHeader->SawOrbitLengthErr() )  meDCCErrorAndWarnConditions_->Fill(dccid,23);
  if (dccHeader->SawTTC_SingErr()    )  meDCCErrorAndWarnConditions_->Fill(dccid,24);
  if (dccHeader->SawTTC_DoubErr()    )  meDCCErrorAndWarnConditions_->Fill(dccid,25);


  ////////// Histogram Spigot Errors from the DCCs HTR Summaries;////////////
  /* [1:8] */ //Histogram HTR Error Bits in the DCC Headers
  bool FoundOne;
  unsigned char WholeErrorList=0; 
  for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
    WholeErrorList=dccHeader->getSpigotErrorBits((unsigned int) j);
    if ((WholeErrorList>>0)&0x01) { //HTR OFW
      ++DCC_DataIntegrityCheck_[bin][15];
      meDCCSummariesOfHTRs_->Fill(dccid, 1);
      fillzoos(11,dccid);
    }
    if ((WholeErrorList>>1)&0x01) { //HTR BSY
      ++DCC_DataIntegrityCheck_[bin][14];
      meDCCSummariesOfHTRs_->Fill(dccid, 2);
      fillzoos(12,dccid);
    }
    if ((WholeErrorList>>2)&0x01) { //EE
      meDCCSummariesOfHTRs_->Fill(dccid, 3);
      mapHTRproblem(dcc_, j);
      fillzoos(1,dccid);
    }
    if ((WholeErrorList>>3)&0x01) { //Trigger Rule Viol.
      meDCCSummariesOfHTRs_->Fill(dccid, 4);
      fillzoos(13,dccid);
    }
    if ((WholeErrorList>>4)&0x01) { //Latency Err
      meDCCSummariesOfHTRs_->Fill(dccid, 5);
    }
    if ((WholeErrorList>>5)&0x01) { //Latency Warn
      meDCCSummariesOfHTRs_->Fill(dccid, 6);
    }
    if ((WholeErrorList>>6)&0x01) { //OD
      meDCCSummariesOfHTRs_->Fill(dccid, 7);
      fillzoos(15,dccid);
    }
    if ((WholeErrorList>>7)&0x01) { //CK
      meDCCSummariesOfHTRs_->Fill(dccid, 8);
      fillzoos(16,dccid);
    }
  }
  /* [9:16] */ //Histogram LRB Error Bits in the DCC Headers
  WholeErrorList=0; 
  for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
    WholeErrorList=dccHeader->getLRBErrorBits((unsigned int) j);
    if ((WholeErrorList>>0)&0x03) { //HammingCode Corrected & Uncorr
      ++DCC_DataIntegrityCheck_[bin][3]; 
      fillzoos(4,dccid);
      mapHTRproblem (dccid, j);
      if ((WholeErrorList>>0)&0x01)  //HammingCode Corrected 
	meDCCSummariesOfHTRs_->Fill(dccid, 9);
      if ((WholeErrorList>>1)&0x01)  //HammingCode Uncorr
	meDCCSummariesOfHTRs_->Fill(dccid, 10);
    }
    for (int i=2; i<8; i++) {
      FoundOne=false;
      if ((WholeErrorList>>i)&0x01) {
	meDCCSummariesOfHTRs_->Fill(dccid, 9+i);
	FoundOne = true;
      }
    }
    if (FoundOne) 
      mapHTRproblem(dcc_, j);
  }
  /* [17:20] */ //Histogram condition of Enabled Spigots without data Present
  bool FoundEnotP=false;
  bool FoundPnotB=false;
  bool FoundPnotV=false;
  bool FoundT=false;
  for(int j=1; j<=HcalDCCHeader::SPIGOT_COUNT; j++) {
    if ( dccHeader->getSpigotEnabled((unsigned int) j-1)         &&
	 (dccHeader->getSpigotDataLength(j-1) <(unsigned long)10) ) 
      ++DCC_DataIntegrityCheck_[bin][8];           // Lost HTR Data for sure
    if (dccHeader->getSpigotEnabled((unsigned int) j-1) &&
	!dccHeader->getSpigotPresent((unsigned int) j-1)      ) FoundEnotP=true;
    //I got the wrong sign on getBxMismatchWithDCC; 
    //It's a match, not a mismatch, when true. I'm sorry. 
    if (dccHeader->getSpigotPresent((unsigned int) j-1) &&
	!dccHeader->getBxMismatchWithDCC((unsigned int) j-1)  ) FoundPnotB=true;
    if (dccHeader->getSpigotPresent((unsigned int) j-1) &&
	!dccHeader->getSpigotValid((unsigned int) j-1)        ) FoundPnotV=true;
    if (dccHeader->getSpigotDataTruncated((unsigned int) j-1) ) {
      ++DCC_DataIntegrityCheck_[bin][7];           // LRB truncated the data
      FoundT=true;}
  }
  if (FoundEnotP)meDCCSummariesOfHTRs_->Fill(dccid,17);
  if (FoundPnotB)meDCCSummariesOfHTRs_->Fill(dccid,18);
  if (FoundPnotV)meDCCSummariesOfHTRs_->Fill(dccid,19);
  if (  FoundT  )meDCCSummariesOfHTRs_->Fill(dccid,20);

  //Fake a problem with each DCC a unique number of times
  // if ((dcc_+1)>= ievt_)
  //   mapDCCproblem(dcc_); 

  // walk through the HTR data...
  HcalHTRData htr;  
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
    if (!dccHeader->getSpigotPresent(spigot)) continue;

    halfhtrDIM_y = 1+(spigot*4);
    chsummDIM_y  = halfhtrDIM_y;
    channDIM_y   = 1+(spigot*3);
    bool chsummAOK=true;
    bool channAOK=true;

    // From this Spigot's DCC header, first.
    WholeErrorList=dccHeader->getLRBErrorBits((unsigned int) spigot);
    if ((WholeErrorList>>0)&0x01)  //HammingCode Corrected 
      ++HalfHTR_DataIntegrityCheck_[halfhtrDIM_x+1][halfhtrDIM_y+2];
    if ((WholeErrorList>>1)&0x01)  //HammingCode Uncorr
      ++HalfHTR_DataIntegrityCheck_[halfhtrDIM_x+1][halfhtrDIM_y+1];
    CRC_err = ((dccHeader->getSpigotSummary(spigot) >> 10) & 0x00000001);
    if (CRC_err)
      ++HalfHTR_DataIntegrityCheck_[halfhtrDIM_x+1][halfhtrDIM_y+0];

    // Load the given decoder with the pointer and length from this spigot.
    dccHeader->getSpigotData(spigot,htr, raw.size()); 
    const unsigned short* HTRraw = htr.getRawData();
    unsigned short HTRwdcount = HTRraw[htr.getRawLength() - 2];
    
    if (HTRwdcount != htr.getRawLength())
      //meChannSumm_DataIntegrityCheck_->Fill(chsummDIM_x, chsummDIM_y);
      ;
    
    //fix me!
    HTRwdcount=htr.getRawLength();

    // Size checks for internal consistency
    // getNTP(), get NDD() seems to be mismatched with format. Manually:
    int NTP = ((htr.getExtHdr6() >> 8) & 0x00FF);
    int NDAQ = (HTRraw[htr.getRawLength() - 4] & 0x7FF);

    if ( !  ((HTRwdcount != 8)               ||
	     (HTRwdcount != 12 + NTP + NDAQ) ||
	     (HTRwdcount != 20 + NTP + NDAQ)    )) {
      ++ChannSumm_DataIntegrityCheck_[chsummDIM_x+1][chsummDIM_y];
      chsummAOK=false;
      //incompatible Sizes declared. Skip it.
      continue; }
    bool EE = ((dccHeader->getSpigotErrorBits(spigot) >> 2) & 0x01);
    if (EE) { 
      if (HTRwdcount != 8) {	//incompatible Sizes declared. Skip it.
	++ChannSumm_DataIntegrityCheck_[chsummDIM_x][chsummDIM_y];
	chsummAOK=false;}
      continue;}
    else{ //For non-EE,
      if ((HTRwdcount-NDAQ-NTP) != 20) {	//incompatible Sizes declared. Skip it.
	++ChannSumm_DataIntegrityCheck_[chsummDIM_x][chsummDIM_y];
	chsummAOK=false; 
	continue;} }

    if (htr.isHistogramEvent()) continue;

    //We trust the data now.  Finish with the check against DCCHeader
    if (htr.getOrbitNumber() != (unsigned int) dccOrN)
      ++HalfHTR_DataIntegrityCheck_[halfhtrDIM_x+0][halfhtrDIM_y+0];
    if (htr.getBunchNumber() != (unsigned int) dccBCN)
      ++HalfHTR_DataIntegrityCheck_[halfhtrDIM_x+0][halfhtrDIM_y+1];
    if (htr.getL1ANumber() != dccEvtNum)
      ++HalfHTR_DataIntegrityCheck_[halfhtrDIM_x+0][halfhtrDIM_y+2];

    bool htrUnSuppressed=(HTRraw[6]>>15 & 0x0001);
    if (htrUnSuppressed) {
      UScount[dcc_][spigot]++;
      int here=1+(HcalDCCHeader::SPIGOT_COUNT*(dcc_))+spigot;
      meUSFractSpigs_->setBinContent(here,
				     ((double)UScount[dcc_][spigot])/(double)ievt_);}

    // Consider removing this check and the histogram it fills; 
    // unless, say, unpacker is a customer (then retitle histo.)
    // check min length, correct wordcount, empty event, or total length if histo event.
    if (!htr.check()) {
      meInvHTRData_ -> Fill(spigot,dccid);
      fillzoos(8,dccid);
      mapHTRproblem(dcc_,spigot);}

    //Fake a problem with each HTR a unique number of times.
    // if ( (spigot+1) >= ievt_ ) 
    //   mapHTRproblem(dcc_,spigot); //Fill every one once, first.


    // Fish out Front-End Errors from the precision channels
    const short unsigned int* daq_first, *daq_last, *tp_first, *tp_last;
    const HcalQIESample* qie_begin, *qie_end, *qie_work;

    // get pointers
    htr.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);

    qie_begin=(HcalQIESample*)daq_first;
    qie_end=(HcalQIESample*)(daq_last+1); // one beyond last..

    int lastcapid=-1;
    int lastfibchan =0, samplecounter=0;
    int channum=0; // Valid: [1,24]
    channDIM_x=0;  

    // Loop over DAQ words for this spigot

    for (qie_work=qie_begin; qie_work!=qie_end; qie_work++) {
      bool yeah = (qie_work->raw()==0xFFFF);
      if (yeah)  // filler word
	continue;
      channAOK=true;
      // Beginning digitized hit of this Half-HTR?
      if (qie_work==qie_begin) {
	channum= (3* (qie_work->fiber() -1)) + qie_work->fiberChan();  
	channDIM_x = (channum*3)+1;
	lastcapid=qie_work->capid();
	samplecounter=1;}
      // or the first TS of a this channel's DAQ data?
      else if (qie_work->fiberAndChan() != lastfibchan) {
	channum= (3* (qie_work->fiber() - 1)) + qie_work->fiberChan();
	channDIM_x = (channum*3)+1;
	//Check the last digi for number of timeslices
	if ((samplecounter != htr.getNDD()) &&
	    (samplecounter != 1)             ) {
	  ++ChannSumm_DataIntegrityCheck_[chsummDIM_x][chsummDIM_y+1];
	  ++Chann_DataIntegrityCheck_[dcc_][channDIM_x][channDIM_y];
	  channAOK=false;}
	samplecounter=1;}
      else { //precision samples not the first timeslice
	int hope = lastcapid +1;
	if (hope==4) hope = 0;
	if (qie_work->capid() != hope){
	  ++ChannSumm_DataIntegrityCheck_[chsummDIM_x+1][chsummDIM_y+1];
	  ++Chann_DataIntegrityCheck_[dcc_][channDIM_x+1][channDIM_y];
	  ++ChannSumm_DataIntegrityCheck_[chsummDIM_x+1][chsummDIM_y+1];
	  ++Chann_DataIntegrityCheck_[dcc_][channDIM_x+1][channDIM_y];
	  channAOK=false;}
	samplecounter++;}
        
      //For every precision data sample in Hcal:

      // FEE - Front End Error
      if (!(qie_work->dv()) || qie_work->er()) {
	++DCC_DataIntegrityCheck_[bin][4]; 
	++ChannSumm_DataIntegrityCheck_[chsummDIM_x+1][chsummDIM_y+2];
	++Chann_DataIntegrityCheck_[dcc_][channDIM_x+1][channDIM_y+1];
	channAOK=false;}
    }

    //Summarize
    if (!channAOK) chsummAOK=false;
    else 
      ++Chann_DataIntegrityCheck_[dcc_][channDIM_x][channDIM_y+1];

    // Prepare for the next round...
    lastcapid=qie_work->capid();
    lastfibchan=qie_work->fiberAndChan();

    if (chsummAOK) //better if every event? Here, every half-HTR's event....
      ++ChannSumm_DataIntegrityCheck_[chsummDIM_x][chsummDIM_y+2];

    if ( !(htr.getErrorsWord() >> 8) & 0x00000001) 
      fillzoos(14,dccid);

    if (dccid==723 && spigot==3) { //the ZDC spigot
      //const unsigned short* zdcRAW =  htr.getRawData();
      //std::cout  << "ZDC ===> " << zdcRAW[0] << std::endl;
    }
    
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    if (prtlvl_ > 0) HTRPrint(htr,prtlvl_);

    unsigned int htrBCN = htr.getBunchNumber(); 
    meBCN_->Fill(htrBCN);
    unsigned int htrEvtN = htr.getL1ANumber();

    if (dccEvtNum != htrEvtN)
      ++DCC_DataIntegrityCheck_[bin][13];
    if ((unsigned int) dccBCN != htrBCN)
      ++DCC_DataIntegrityCheck_[bin][11];
 
    unsigned int fib1BCN = htr.getFib1OrbMsgBCN();
    unsigned int fib2BCN = htr.getFib2OrbMsgBCN();
    unsigned int fib3BCN = htr.getFib3OrbMsgBCN();
    unsigned int fib4BCN = htr.getFib4OrbMsgBCN();
    unsigned int fib5BCN = htr.getFib5OrbMsgBCN();
    unsigned int fib6BCN = htr.getFib6OrbMsgBCN();
    unsigned int fib7BCN = htr.getFib7OrbMsgBCN();
    unsigned int fib8BCN = htr.getFib8OrbMsgBCN();
    
    if ( (dccEvtNum != fib1BCN) ||
	 (dccEvtNum != fib2BCN) ||
	 (dccEvtNum != fib3BCN) ||
	 (dccEvtNum != fib4BCN) ||
	 (dccEvtNum != fib5BCN) ||
	 (dccEvtNum != fib6BCN) ||
	 (dccEvtNum != fib7BCN) ||
	 (dccEvtNum != fib8BCN) ) 
      ++DCC_DataIntegrityCheck_[bin][10];
 
    meFibBCN_->Fill(fib1BCN);
    meFibBCN_->Fill(fib2BCN);
    meFibBCN_->Fill(fib3BCN);
    meFibBCN_->Fill(fib4BCN);
    meFibBCN_->Fill(fib5BCN);
    meFibBCN_->Fill(fib6BCN);
    meFibBCN_->Fill(fib7BCN);
    meFibBCN_->Fill(fib8BCN);
    /* Disable for now
       meFib1OrbMsgBCN_->Fill(slotnum, cratenum, fib1BCN);
       meFib2OrbMsgBCN_->Fill(slotnum, cratenum, fib2BCN);
       meFib3OrbMsgBCN_->Fill(slotnum, cratenum, fib3BCN);
       meFib4OrbMsgBCN_->Fill(slotnum, cratenum, fib4BCN);
       meFib5OrbMsgBCN_->Fill(slotnum, cratenum, fib5BCN);
       meFib6OrbMsgBCN_->Fill(slotnum, cratenum, fib6BCN);
       meFib7OrbMsgBCN_->Fill(slotnum, cratenum, fib7BCN);
       meFib8OrbMsgBCN_->Fill(slotnum, cratenum, fib8BCN);
    */  

    unsigned int htrFWVer = htr.getFirmwareRevision() & 0xFF;
    meFWVersion_->Fill(cratenum,htrFWVer);

    ///check that all HTRs have the same L1A number
    unsigned int refEvtNum = dccEvtNum;
    /*Could use Evt # from dcc as reference, but not now.
      if(htr.getL1ANumber()!=refEvtNum) {meEvtNumberSynch_->Fill(slotnum,cratenum);
      if (prtlvl_ == 1)cout << "++++ Evt # out of sync, ref, this HTR: "<< refEvtNum << "  "<<htr.getL1ANumber() <<endl;
      }
    */

    if(lastEvtN_==-1) {lastEvtN_ = htrEvtN;  ///the first one will be the reference
    refEvtNum = lastEvtN_;
    int EvtNdiff = htrEvtN - dccEvtNum;
    meEvtNCheck_->Fill(EvtNdiff);
    }
    else {
      if((int) htrEvtN!=lastEvtN_) {meEvtNumberSynch_->Fill(slotnum,cratenum);
      if (prtlvl_ == 1)cout << "++++ Evt # out of sync, ref, this HTR: "<< lastEvtN_ << "  "<<htrEvtN <<endl;}

    }

    ///check that all HTRs have the same BCN

    unsigned int refBCN = dccBCN;
    /*Could use BCN from dcc as reference, but not now.
      if(htr.getBunchNumber()!=refBCN) {meBCNSynch_->Fill(slotnum,cratenum);
      if (prtlvl_==1)cout << "++++ BCN # out of sync, ref, this HTR: "<< refBCN << "  "<<htrBCN <<endl;
      }
    */
    // Use 1st HTR as reference
    if(lastBCN_==-1) {lastBCN_ = htrBCN;  ///the first one will be the reference
    refBCN = lastBCN_;
    int BCNdiff = htrBCN-dccBCN;
    meBCNCheck_->Fill(BCNdiff);
    }

    else {
      if((int)htrBCN!=lastBCN_) {meBCNSynch_->Fill(slotnum,cratenum);
      if (prtlvl_==1)cout << "++++ BCN # out of sync, ref, this HTR: "<< lastBCN_ << "  "<<htrBCN <<endl;}
    }

    MonitorElement* tmpErr = 0;
    bool valid = false;
    for(int fchan=0; fchan<3 && !valid; fchan++){
      for(int fib=0; fib<9 && !valid; fib++){
	HcalElectronicsId eid(fchan,fib,spigot,dccid-firstFED_);
	eid.setHTR(htr.readoutVMECrateId(),htr.htrSlot(),htr.htrTopBottom());
	DetId did=emap.lookup(eid);
	//	if (did.null()){cout << " Detector id null  " << cratenum << "  " <<slotnum << endl;}
	if (!did.null()) {
	  switch (((HcalSubdetector)did.subdetId())) {
	  case (HcalBarrel): {
	    tmpErr = DCC_ErrWd_HBHE;
	    valid = true;
	  } break;
	  case (HcalEndcap): {
	    tmpErr = DCC_ErrWd_HBHE;
	    valid = true;
	  } break;
	  case (HcalOuter): {
	    tmpErr = DCC_ErrWd_HO;
	    valid = true;
	  } break;
	  case (HcalForward): {
	    tmpErr = DCC_ErrWd_HF; 
	    valid = true;
	  } break;
	  default: break;
	  }
	}
      }
    }    
    int errWord = htr.getErrorsWord() & 0x1FFFF;
    if(tmpErr!=NULL){
      for(int i=0; i<16; i++){
	int errbit = errWord&(0x01<<i);
	// Bit 15 should always be 1; consider it an error if it isn't.
	if (i==15) errbit = errbit - 0x8000;
	if (errbit !=0){
	  tmpErr->Fill(i);
	  if (i==5 && prtlvl_ != -1) continue; // Skip latency warning for now
	  meErrWdCrate_->Fill(cratenum,i);
	  if (cratenum ==0)meCrate0HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==1)meCrate1HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==2)meCrate2HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==3)meCrate3HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==4)meCrate4HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==5)meCrate5HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==6)meCrate6HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==7)meCrate7HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==8)meCrate8HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==9)meCrate9HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==10)meCrate10HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==11)meCrate11HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==12)meCrate12HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==13)meCrate13HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==14)meCrate14HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==15)meCrate15HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==16)meCrate16HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==17)meCrate17HTRErr_ -> Fill(slotnum,i);
	} 
      }
    }    
  } //  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) 
  return;
} // void HcalDataFormatMonitor::unpack(

void HcalDataFormatMonitor::HTRPrint(const HcalHTRData& htr,int prtlvl){


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
void HcalDataFormatMonitor::labelHTRBits(MonitorElement* mePlot,unsigned int axisType) {

  if (axisType !=1 && axisType != 2) return;

  mePlot -> setBinLabel(1,"Overflow Warn",axisType);
  mePlot -> setBinLabel(2,"Buffer Busy",axisType);
  mePlot -> setBinLabel(3,"Empty Event",axisType);
  mePlot -> setBinLabel(4,"Reject L1A",axisType);
  mePlot -> setBinLabel(5,"Latency Err",axisType);
  mePlot -> setBinLabel(6,"Latency Warn",axisType);
  mePlot -> setBinLabel(7,"OptDat Err",axisType);
  mePlot -> setBinLabel(8,"Clock Err",axisType);
  mePlot -> setBinLabel(9,"Bunch Err",axisType);
  mePlot -> setBinLabel(10,"Link Err",axisType);
  mePlot -> setBinLabel(11,"CapId Err",axisType);
  mePlot -> setBinLabel(12,"FE Format Err",axisType);
  mePlot -> setBinLabel(13,"Test Mode",axisType);
  mePlot -> setBinLabel(14,"Histo Mode",axisType);
  mePlot -> setBinLabel(15,"Calib Trig",axisType);
  mePlot -> setBinLabel(16,"Bit15 Err",axisType);

  return;
}

//No size checking; better have enough y-axis bins!
void HcalDataFormatMonitor::label_ySpigots(MonitorElement* me_ptr, int ybins) {
  char label[10];
  for (int spig=0; spig<HcalDCCHeader::SPIGOT_COUNT; spig++) {
    sprintf(label, "Spgt %02d", spig);
    me_ptr->setBinLabel((2+(spig*ybins)), label, 2); //margin of 1 at low value
  }
}

//No size checking; better have enough x-axis bins!
void HcalDataFormatMonitor::label_xChanns(MonitorElement* me_ptr, int xbins) {
  char label[10];
  for (int ch=0; ch<HcalHTRData::CHANNELS_PER_SPIGOT; ch++) {
    sprintf(label, "Ch %02d", ch+1);
    me_ptr->setBinLabel((2+(ch*xbins)), label, 1); //margin of 3 at low value
  }
}

//No size checking; better have enough x-axis bins!
void HcalDataFormatMonitor::label_xFEDs(MonitorElement* me_ptr, int xbins) {
  char label[10];
  for (int spig=0; spig<32; spig++) {
    sprintf(label, "%03d", spig+700);
    me_ptr->setBinLabel((2+(spig*xbins)), label, 1); //margin of 1 at low value
  }
}


void HcalDataFormatMonitor::labelthezoo(MonitorElement* zoo) {
  ///\\\///  zoo-> setBinLabel(1,"EE - HTR lost",1);    //HTR sent Empty Event
  ///\\\///  zoo-> setBinLabel(2,"LRB lost",1);	    //LRB truncated data
  ///\\\///  zoo-> setBinLabel(3,"FEE",1);		    //CapID or DV not set correctly
  ///\\\///  zoo-> setBinLabel(4,"HAM",1);		    //Corr. & Uncorr. Hamm.
  ///\\\///  zoo-> setBinLabel(5,"HTR CRC",1);	    //DCC found HTR CRC err
  ///\\\///  zoo-> setBinLabel(6,"CMS EvFmt",1);	    //Common Data Format failed check
  ///\\\///  zoo-> setBinLabel(7,"DCC EvFmt",1);	    //DCC Event format failed check
  ///\\\///  zoo-> setBinLabel(8,"HTR EvFmt",1);	    //HTR Event Format failed check
  ///\\\///  zoo-> setBinLabel(9,"DCC OFW",1);	    //DCC Overflow Warning
  ///\\\///  zoo-> setBinLabel(10,"DCC BSY",1);	    //DCC Busy 
  ///\\\///  zoo-> setBinLabel(11,"HTR OFW",1);	    //HTR Overflow Warning
  ///\\\///  zoo-> setBinLabel(12,"HTR BSY",1);	    //HTR Busy 
  ///\\\///  zoo-> setBinLabel(13,"RL1A",1);	    //HTR Rejected L1A
  ///\\\///  zoo-> setBinLabel(14,"BE",1);		    //HTR Bunchcount Error
  ///\\\///  zoo-> setBinLabel(15,"OD",1);		    //Optical Data error  
  ///\\\///  zoo-> setBinLabel(16,"CK",1);		    //Clock Error from HTR              
}

// Public function so HcalMonitorModule can slip in a 
// logical map digest or two. 
void HcalDataFormatMonitor::smuggleMaps(std::map<uint32_t, std::vector<HcalDetId> >& givenDCCtoCell,
					std::map<pair <int,int> , std::vector<HcalDetId> >& givenHTRtoCell) {
  DCCtoCell = givenDCCtoCell;
  HTRtoCell = givenHTRtoCell;
  return;
}

void HcalDataFormatMonitor::fillzoos(int bin, int dccid) {
  ///\\\///  DATAFORMAT_PROBLEM_ZOO->Fill(bin);
  ///\\\///  if (HBHE_LO_DCC<=dccid && dccid<=HBHE_HI_DCC)
  ///\\\///    HBHE_DATAFORMAT_PROBLEM_ZOO->Fill(bin);
  ///\\\///  if (HF_LO_DCC<=dccid && dccid<=HF_HI_DCC)
  ///\\\///    HF_DATAFORMAT_PROBLEM_ZOO->Fill(bin);
  ///\\\///  if (HO_LO_DCC<=dccid && dccid<=HO_HI_DCC)
  ///\\\///    HO_DATAFORMAT_PROBLEM_ZOO->Fill(bin);
}

void HcalDataFormatMonitor::mapHTRproblem (int dcc, int spigot) {
  int mydepth,myeta = 0;
  //dcc, spigot pair for finding this spigot's HcalDetIds
  pair <int,int> thishtr = pair <int,int> (dcc, spigot);

  //Light up all affected cells.
  for (std::vector<HcalDetId>::iterator thishdi = HTRtoCell[thishtr].begin(); 
       thishdi != HTRtoCell[thishtr].end(); thishdi++) {
    mydepth = thishdi->depth() - 1; //Array indexes in mydepth, cells are labeled in depth. Sigh....

    if (thishdi->subdet() == HcalForward) {
      if (thishdi->zside() < 0)  myeta   = thishdi->ieta()+int((etaBins_-2)/2)-1;
      else                       myeta   = thishdi->ieta()+int((etaBins_-2)/2)+1;
    } else {                     myeta   = thishdi->ieta()+int((etaBins_-2)/2); }
    problemfound[myeta][thishdi->iphi()-1][mydepth] = true;
  }
}   

void HcalDataFormatMonitor::mapDCCproblem(int dcc) {
  int mydepth,myeta = 0;

  //Light up all affected cells.
  for (std::vector<HcalDetId>::iterator thishdi = DCCtoCell[dcc].begin(); 
       thishdi != DCCtoCell[dcc].end(); thishdi++) {
    mydepth = thishdi->depth() - 1; //Array indexes in mydepth, cells are labeled in depth. Sigh....
    if ( (thishdi->subdet() == HcalForward )   ) {
      if (thishdi->zside() < 0)  myeta   = thishdi->ieta()+int((etaBins_-2)/2)-1;
      else                       myeta   = thishdi->ieta()+int((etaBins_-2)/2)+1;
    } else {                     myeta   = thishdi->ieta()+int((etaBins_-2)/2); }
    problemfound[myeta][thishdi->iphi()-1][mydepth] = true;
  }
}

void HcalDataFormatMonitor::UpdateMEs (void ) {
  for (int x=0; x<RCDIX; x++)
    for (int y=0; y<RCDIY; y++)
      if (DCC_DataIntegrityCheck_[x][y]) //If it's not zero
	meDCC_DataIntegrityCheck_->Fill(x, y, DCC_DataIntegrityCheck_[x][y]);

  for (int x=0; x<HHDIX; x++)
    for (int y=0; y<HHDIY; y++)
      if (HalfHTR_DataIntegrityCheck_  [x][y])
	meHalfHTR_DataIntegrityCheck_->Fill(x,y,HalfHTR_DataIntegrityCheck_[x][y]);
  	 
  for (int x=0; x<CSDIX; x++)
    for (int y=0; y<HHDIY; y++)
      if (ChannSumm_DataIntegrityCheck_[x][y])
	meChannSumm_DataIntegrityCheck_->Fill(x,y,ChannSumm_DataIntegrityCheck_[x][y]);

  for (int f=0; f<NUMDCCS; f++)
    for (int x=0; x<CIX; x++)
      for (int y=0; y<CIY; y++)      
	if (Chann_DataIntegrityCheck_[f][x][y])
	  meChann_DataIntegrityCheck_[f]->Fill(x,y,Chann_DataIntegrityCheck_ [f][x][y]);

  uint64_t probfrac=0;
  int ieta=0;
  int iphi=0;

  for (int eta=0;eta<(etaBins_-2);eta++) {
    ieta=eta-(int)ceil((etaBins_-2)/2);
    for (int phi=0;phi<PHIBINS;++phi) {
      iphi=phi+1;
      for (int depth=0;depth<DEPTHBINS;++depth) {// this is one unit less "true" depth (for indexing purposes)
	if (0 == problemcount[eta][phi][depth]) 
	  continue;
	  // remember that HF's elements are stored in towers farther forward than "true"
        if ( ((depth==0)||(depth==1)) &&
	     (abs(ieta)>29)              ) { 
	  if (eta<int((etaBins_-2)/2)) ieta=eta-int((etaBins_-2)/2)+1;
	  else                         ieta=eta-int((etaBins_-2)/2)-1;
	} // else Not HcalForward
	probfrac = ((uint64_t) problemcount[eta][phi][depth] ); // / (uint64_t) ievt_);   
	//Select HcalEndcap d1,2 while sidestepping HcalForward
	if ( ( (depth==0) || (depth==1) )               &&
	     ( abs(eta-(int)ceil((etaBins_-2)/2)) <=29) &&  
	     ( abs(eta-(int)ceil((etaBins_-2)/2)) >=17)    ) {
	  //Bump apart HEd1,2 for the StJ6
	  HWProblemsByDepth_[depth+4]->setBinContent(ieta+((etaBins_-2)/2)+2, iphi+1, probfrac);
	} else{
	  HWProblemsByDepth_[depth  ]->setBinContent(ieta+((etaBins_-2)/2)+2, iphi+1, probfrac);
	}
      } //depth
    } //phi
  } //eta
} //UpdateMEs
