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

  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      HalfHTRDataCorruptionIndicators_  [x][y]=0;

  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      LRBDataCorruptionIndicators_  [x][y]=0;
  	 
  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<TWO__SPGT; y++)
      ChannSumm_DataIntegrityCheck_[x][y]=0;

  for (int f=0; f<NUMDCCS; f++)
    for (int x=0; x<  TWO_CHANN; x++)
      for (int y=0; y<TWO__SPGT; y++)      
	Chann_DataIntegrityCheck_  [f][x][y]=0;

  for (int i=0; i<(NUMDCCS * NUMSPIGS * HTRCHANMAX); i++) 
    hashedHcalDetId_[i]=HcalDetId::Undefined;

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
  
  baseFolder_ = rootFolder_+"DataFormatMonitor";

  if(fVerbosity) cout << "About to pushback fedUnpackList_" << endl;

  firstFED_ = FEDNumbering::MINHCALFEDID;
  for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++) 
    {
      if(fVerbosity>1) cout << "[DFMon:]Pushback for fedUnpackList_: " << i <<endl;
      fedUnpackList_.push_back(i);
    }

  prtlvl_ = ps.getUntrackedParameter<int>("dfPrtLvl");
  dfmon_checkNevents = ps.getUntrackedParameter<int>("DataFormatMonitor_checkNevents",checkNevents_);
  AllowedCalibTypes_ = ps.getUntrackedParameter<vector<int> >("DataFormatMonitor_AllowedCalibTypes",AllowedCalibTypes_);

  if (m_dbe)
    {
      std::string type;
      m_dbe->setCurrentFolder(baseFolder_);
      
      ProblemCells=m_dbe->book2D(" HardwareWatchCells",
				 " Hardware Watch Cells for HCAL",
				 85,-42.5,42.5,
				 72,0.5,72.5);
      ProblemCells->setAxisTitle("i#eta",1);
      ProblemCells->setAxisTitle("i#phi",2);
      SetEtaPhiLabels(ProblemCells);
      SetupEtaPhiHists(ProblemCellsByDepth," Hardware Watch Cells", "");
      
      //Initialize maps "problemcount" and "problemfound" before first event.
      unsigned int etabins=0;
      unsigned int phibins=0;
      for (unsigned int depth=0; depth<4; ++depth)
	{
	  etabins=ProblemCellsByDepth.depth[depth]->getNbinsX();
	  phibins=ProblemCellsByDepth.depth[depth]->getNbinsY();
	  for (unsigned int eta=0; eta<etabins;++eta)
	    {
	      for (unsigned int phi=0;phi<phibins;++phi)
		{
		  problemcount[eta][phi][depth]=0;
		  problemfound[eta][phi][depth]=false;
		}
	    }
	}
      
      meEVT_ = m_dbe->bookInt("Data Format Task Event Number");
      meEVT_->Fill(ievt_);    
      meTOTALEVT_ = m_dbe->bookInt("Data Format Total Events Processed");
      meTOTALEVT_->Fill(tevt_);
      
      m_dbe->setCurrentFolder(baseFolder_ + "/Corruption"); /// Below, "Corruption" FOLDER
      type = "01 Common Data Format violations";
      meCDFErrorFound_ = m_dbe->book2D(type,type,32,699.5,731.5,9,0.5,9.5);
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
      meDCCEventFormatError_ = m_dbe->book2D(type,type,32,699.5,731.5,6,0.5,6.5);
      meDCCEventFormatError_->setAxisTitle("HCAL FED ID", 1);
      meDCCEventFormatError_->setBinLabel(1, "FmtVers Changed", 2);
      meDCCEventFormatError_->setBinLabel(2, "StrayBits Changed", 2);
      meDCCEventFormatError_->setBinLabel(3, "HTRStatusPad", 2);
      meDCCEventFormatError_->setBinLabel(4, "32bitPadErr", 2);
      meDCCEventFormatError_->setBinLabel(5, "Number Mismatch Bit Miscalc", 2);      
      meDCCEventFormatError_->setBinLabel(6, "Low 8 HTR Status Bits Miscopy", 2);	       
      
      type = "04 HTR BCN when OrN Diff";
      meBCNwhenOrNDiff_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
      meBCNwhenOrNDiff_->setAxisTitle("BCN",1);
      meBCNwhenOrNDiff_->setAxisTitle("# of Entries",2);
      
      type = "03 OrN Difference HTR - DCC";
      meOrNCheck_ = m_dbe->book1D(type,type,65,-32.5,32.5);
      meOrNCheck_->setAxisTitle("htr OrN - dcc OrN",1);
      
      type = "03 OrN Inconsistent - HTR vs DCC";
      meOrNSynch_= m_dbe->book2D(type,type,32,0,32, 15,0,15);
      meOrNSynch_->setAxisTitle("FED #",1);
      meOrNSynch_->setAxisTitle("Spigot #",2);
      
      type = "05 BCN Difference HTR - DCC";
      meBCNCheck_ = m_dbe->book1D(type,type,501,-250.5,250.5);
      meBCNCheck_->setAxisTitle("htr BCN - dcc BCN",1);
      
      type = "05 BCN Inconsistent - HTR vs DCC";
      meBCNSynch_= m_dbe->book2D(type,type,32,0,32, 15,0,15);
      meBCNSynch_->setAxisTitle("FED #",1);
      meBCNSynch_->setAxisTitle("Slot #",2);
      
      type = "06 EvN Difference HTR - DCC";
      meEvtNCheck_ = m_dbe->book1D(type,type,601,-300.5,300.5);
      meEvtNCheck_->setAxisTitle("htr Evt # - dcc Evt #",1);
      
      type = "06 EvN Inconsistent - HTR vs DCC";
      meEvtNumberSynch_= m_dbe->book2D(type,type,32,0,32, 15,0,15);
      meEvtNumberSynch_->setAxisTitle("FED #",1);
      meEvtNumberSynch_->setAxisTitle("Slot #",2);
      
      //     ---------------- 
      //     | E!P | UE | TR |                                           
      // ----|  ND | OV | ID |					       
      // | T | CRC | ST | ODD| 					       
      // -------------------- 					       
      type="07 LRB Data Corruption Indicators";  
      meLRBDataCorruptionIndicators_= m_dbe->book2D(type,type,
						    THREE_FED,0,THREE_FED,
						    THREE_SPG,0,THREE_SPG);
      label_xFEDs   (meLRBDataCorruptionIndicators_, 4); // 3 bins + 1 margin per ch.
      label_ySpigots(meLRBDataCorruptionIndicators_, 4); // 3 bins + 1 margin each spgt
      
      //     ---------------- 
      //     | CT | BE | LW |
      //     | HM | 15 | WW | (Wrong Wordcount)
      //     | TM | CK | IW | (Illegal Wordcount)
      //     ---------------- 
      type="08 Half-HTR Data Corruption Indicators";
      meHalfHTRDataCorruptionIndicators_= m_dbe->book2D(type,type,
							THREE_FED,0,THREE_FED,
							THREE_SPG,0,THREE_SPG);
      label_xFEDs   (meHalfHTRDataCorruptionIndicators_, 4); // 3 bins + 1 margin per ch.
      label_ySpigots(meHalfHTRDataCorruptionIndicators_, 4); // 3 bins + 1 margin each spgt
      
      //    ------------
      //    | !DV | Er  |
      //    | NTS | Cap |
      //    ------------
      type = "09 Channel Integrity Summarized by Spigot";
      meChannSumm_DataIntegrityCheck_= m_dbe->book2D(type,type,
						     TWO___FED,0,TWO___FED,
						     TWO__SPGT,0,TWO__SPGT);
      label_xFEDs   (meChannSumm_DataIntegrityCheck_, 3); // 2 bins + 1 margin per ch.
      label_ySpigots(meChannSumm_DataIntegrityCheck_, 3); // 2 bins + 1 margin per spgt
      
      m_dbe->setCurrentFolder(baseFolder_ + "/Corruption/Channel Data Integrity");
      char label[10];
      for (int f=0; f<NUMDCCS; f++){      
	sprintf(label, "FED %03d Channel Integrity", f+700);
	meChann_DataIntegrityCheck_[f] =  m_dbe->book2D(label,label,
							TWO_CHANN,0,TWO_CHANN,
							TWO__SPGT,0,TWO__SPGT);
	label_xChanns (meChann_DataIntegrityCheck_[f], 3); // 2 bins + 1 margin per ch.
	label_ySpigots(meChann_DataIntegrityCheck_[f], 3); // 2 bins + 1 margin per spgt
	;}
      
      m_dbe->setCurrentFolder(baseFolder_ + "/Data Flow"); ////Below, "Data Flow" FOLDER
      type="DCC Event Counts";
      mefedEntries_ = m_dbe->book1D(type,type,32,699.5,731.5);
      
      type = "BCN from DCCs";
      medccBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
      medccBCN_->setAxisTitle("BCN",1);
      medccBCN_->setAxisTitle("# of Entries",2);
      
      type = "BCN from HTRs";
      meBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
      meBCN_->setAxisTitle("BCN",1);
      meBCN_->setAxisTitle("# of Entries",2);
      
      type = "DCC Data Block Size Distribution";
      meFEDRawDataSizes_=m_dbe->book1D(type,type,1200,-0.5,12000.5);
      meFEDRawDataSizes_->setAxisTitle("# of bytes",1);
      meFEDRawDataSizes_->setAxisTitle("# of Data Blocks",2);
      
      type = "DCC Data Block Size Profile";
      meEvFragSize_ = m_dbe->bookProfile(type,type,32,699.5,731.5,100,-1000.0,12000.0,"");
      type = "DCC Data Block Size Each FED";
      meEvFragSize2_ =  m_dbe->book2D(type,type,64,699.5,731.5, 240,0,12000);
      
      //     ------------
      //     | OW | OFW |    "Two Caps HTR; Three Caps FED."
      //     | BZ | BSY |
      //     | EE | RL  |
      // ----------------
      // | CE |            (corrected error, Hamming code)
      // ------
      type = "01 Data Flow Indicators";
      meDataFlowInd_= m_dbe->book2D(type,type,
				    TWO___FED,0,TWO___FED,
				    THREE_SPG,0,THREE_SPG);
      label_xFEDs   (meDataFlowInd_, 3); // 2 bins + 1 margin per ch.
      label_ySpigots(meDataFlowInd_, 4); // 3 bins + 1 margin each spgt
      
      m_dbe->setCurrentFolder(baseFolder_ + "/Diagnostics"); ////Below, "Diagnostics" FOLDER
      
      type = "DCC Firmware Version";
      meDCCVersion_ = m_dbe->bookProfile(type,type, 32, 699.5, 731.5, 256, -0.5, 255.5);
      meDCCVersion_ ->setAxisTitle("FED ID", 1);
      
      type = "HTR Status Word HBHE";
      HTR_StatusWd_HBHE =  m_dbe->book1D(type,type,16,-0.5,15.5);
      labelHTRBits(HTR_StatusWd_HBHE,1);
      
      type = "HTR Status Word HF";
      HTR_StatusWd_HF =  m_dbe->book1D(type,type,16,-0.5,15.5);
      labelHTRBits(HTR_StatusWd_HF,1);
      
      type = "HTR Status Word HO";
      HTR_StatusWd_HO = m_dbe->book1D(type,type,16,-0.5,15.5);
      labelHTRBits(HTR_StatusWd_HO,1);
      
      int maxbits = 16;//Look at all 16 bits of the Error Words
      type = "HTR Status Word by Crate";
      meErrWdCrate_ = m_dbe->book2D(type,type,18,-0.5,17.5,maxbits,-0.5,maxbits-0.5);
      meErrWdCrate_ -> setAxisTitle("Crate #",1);
      labelHTRBits(meErrWdCrate_,2);
      
      type = "Unpacking - HcalHTRData check failures";
      meInvHTRData_= m_dbe->book2D(type,type,16,-0.5,15.5,32,699.5,731.5);
      meInvHTRData_->setAxisTitle("Spigot #",1);
      meInvHTRData_->setAxisTitle("DCC #",2);
      
      type = "HTR Fiber Orbit Message BCN";
      meFibBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
      meFibBCN_->setAxisTitle("BCN of Fib Orb Msg",1);
      
      type = "HTR Status Word - Crate 0";
      meCrate0HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate0HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate0HTRErr_,2);
      
      type = "HTR Status Word - Crate 1";
      meCrate1HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate1HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate1HTRErr_,2);
      
      type = "HTR Status Word - Crate 2";
      meCrate2HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate2HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate2HTRErr_,2);
      
      type = "HTR Status Word - Crate 3";
      meCrate3HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate3HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate3HTRErr_,2);
      
      type = "HTR Status Word - Crate 4";
      meCrate4HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate4HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate4HTRErr_,2);

      type = "HTR Status Word - Crate 5";
      meCrate5HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate5HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate5HTRErr_,2);

      type = "HTR Status Word - Crate 6";
      meCrate6HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate6HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate6HTRErr_,2);

      type = "HTR Status Word - Crate 7";
      meCrate7HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate7HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate7HTRErr_,2);

      type = "HTR Status Word - Crate 9";
      meCrate9HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate9HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate9HTRErr_,2);

      type = "HTR Status Word - Crate 10";
      meCrate10HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate10HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate10HTRErr_,2);

      type = "HTR Status Word - Crate 11";
      meCrate11HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate11HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate11HTRErr_,2);

      type = "HTR Status Word - Crate 12";
      meCrate12HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate12HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate12HTRErr_,2);

      type = "HTR Status Word - Crate 13";
      meCrate13HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate13HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate13HTRErr_,2);

      type = "HTR Status Word - Crate 14";
      meCrate14HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate14HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate14HTRErr_,2);

      type = "HTR Status Word - Crate 15";
      meCrate15HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate15HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate15HTRErr_,2);

      type = "HTR Status Word - Crate 17";
      meCrate17HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
      meCrate17HTRErr_ ->setAxisTitle("Slot #",1);
      labelHTRBits(meCrate17HTRErr_,2);

      type = "HTR UnSuppressed Event Fractions";
      meUSFractSpigs_ = m_dbe->book1D(type,type,481,0,481);
      for(int f=0; f<NUMDCCS; f++) {
	sprintf(label, "FED 7%02d", f);
	meUSFractSpigs_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f), label);
	for(int s=1; s<HcalDCCHeader::SPIGOT_COUNT; s++) {
	  sprintf(label, "sp%02d", s);
	  meUSFractSpigs_->setBinLabel(1+(HcalDCCHeader::SPIGOT_COUNT*f)+s, label);}}

      // Firmware version
      type = "HTR Firmware Version";
      //  Maybe change to Profile histo eventually
      //meHTRFWVersion_ = m_dbe->bookProfile(type,type,18,-0.5,17.5,245,10.0,255.0,"");
      meHTRFWVersion_ = m_dbe->book2D(type,type ,18,-0.5,17.5,180,75.5,255.5);
      meHTRFWVersion_->setAxisTitle("Crate #",1);
      meHTRFWVersion_->setAxisTitle("HTR Firmware Version",2);

      type = "HTR Fiber 1 Orbit Message BCNs";
      meFib1OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 2 Orbit Message BCNs";
      meFib2OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 3 Orbit Message BCNs";
      meFib3OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 4 Orbit Message BCNs";
      meFib4OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 5 Orbit Message BCNs";
      meFib5OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 6 Orbit Message BCNs";
      meFib6OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 7 Orbit Message BCNs";
      meFib7OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      type = "HTR Fiber 8 Orbit Message BCNs";
      meFib8OrbMsgBCN_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    } // if (m_dbe)
  return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw, 
					 const HcalUnpackerReport& report, 
					 const HcalElectronicsMap& emap,
					 int CalibType){
  
  if(!m_dbe) { 
    printf("HcalDataFormatMonitor::processEvent DQMStore not instantiated!!!\n");  
    return;}
  
  bool processevent=false;
  if (AllowedCalibTypes_.size()==0)
    processevent=true;
  else
    {
      for (unsigned int i=0;i<AllowedCalibTypes_.size();++i)
        {
          if (AllowedCalibTypes_[i]==CalibType)
            {
              processevent=true;
              break;
            }
        }
    }
  if (fVerbosity>1) std::cout <<"<HcalDataFormatMonitor::processEvent>  calibType = "<<CalibType<<"  processing event? "<<processevent<<endl;
  if (!processevent)
    return;


  HcalBaseMonitor::processEvent();
  
  // Call these to make sure histograms get updated
  ProblemCells->update();
  for (int depth=0;depth<4;++depth) 
    ProblemCellsByDepth.depth[depth]->update();

  lastEvtN_ = -1;
  lastBCN_ = -1;
  lastOrN_ = -1;
  
  // Fill event counters (underflow bins of histograms)
  // This is the only way we can make these histograms appear in online DQM!
  // Weird!  -- Jeff, 4/27/09
  meLRBDataCorruptionIndicators_->update();
  meHalfHTRDataCorruptionIndicators_->update();
  meChannSumm_DataIntegrityCheck_->update();
  for (int f=0; f<NUMDCCS; f++)      
    meChann_DataIntegrityCheck_[f]->update();
  meDataFlowInd_->update();

  // Loop over all FEDs reporting the event, unpacking if good.
  for (vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    if (fed.size()<12) continue;  //At least the size of headers and trailers of a DCC.
    unpack(fed,emap); //Interpret data, fill histograms, everything.
  }

  // Any problem worth mapping, anywhere?
  unsigned int etabins=0;
  unsigned int phibins=0;
  for (unsigned int depth=0; depth<4; ++depth)
    {
      etabins=ProblemCellsByDepth.depth[depth]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[depth]->getNbinsY();
      for (unsigned int eta=0; eta<etabins;++eta)
	{
	  for (unsigned int phi=0;phi<phibins;++phi)
	    {
	      if (problemfound[eta][phi][depth])
		++problemcount[eta][phi][depth];
	    }
	}
    }

  //if (0== (ievt_ % dfmon_checkNevents))
  //  UpdateMEs();
  //Transfer this event's problem info to 
  for (unsigned int depth=0; depth<4; ++depth)
    {
      etabins=ProblemCellsByDepth.depth[depth]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[depth]->getNbinsY();
      for (unsigned int eta=0; eta<etabins;++eta)
	{
	  for (unsigned int phi=0;phi<phibins;++phi)
	    {
	      problemfound[eta][phi][depth]=false;		
	    }
	}
    }
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
  if(fVerbosity>1) cout << "DCC " << dccid << endl;
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
    CDFReservedBits_list.insert(pair<int,short>
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
    if (fVerbosity>0) cout <<"CDFProbThisDCC"<<endl;
    mapDCCproblem(dcc_);
  }

  mefedEntries_->Fill(dccid);

  CDFProbThisDCC = false;  // reset for the next go-round.
  
  char CRC_err;
  for(int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    CRC_err = ((dccHeader->getSpigotSummary(i) >> 10) & 0x00000001);
    if (CRC_err) {
      //Set the problem flag for the ieta, iphi of any channel in this DCC
      if (fVerbosity>0) cout <<"HTR Problem: CRC_err"<<endl;
      mapHTRproblem(dcc_, i);  
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
    DCCEvtFormat_list.insert(pair<int,short>
			     (dccid,dccHeader->getDCCDataFormatVersion() ) );
    DCCEvtFormat_it = DCCEvtFormat_list.find(dccid);
  } // then check against it.
  if (dccHeader->getDCCDataFormatVersion()!= DCCEvtFormat_it->second) {
    meDCCEventFormatError_->Fill(dccid,1);
    if (fVerbosity>0) cout <<"DCC Error Type 1"<<endl;
    mapDCCproblem(dcc_);
  }
  /* 2 */ //Check for ones where there should always be zeros
  if (false) //dccHeader->getByte1Zeroes() || dccHeader->getByte3Zeroes() || dccHeader->getByte567Zeroes()) 
  {
    meDCCEventFormatError_->Fill(dccid,2);
    if (fVerbosity>0) cout <<"DCC Error Type 2"<<endl;
    mapDCCproblem(dcc_);
  }
  /* 3 */ //Check that there are zeros following the HTR Status words.
  int SpigotPad = HcalDCCHeader::SPIGOT_COUNT;
  if (  (((uint64_t) dccHeader->getSpigotSummary(SpigotPad)  ) 
	 | ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+1)) 
	 | ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+2)))  != 0){
    meDCCEventFormatError_->Fill(dccid,3);
  if (fVerbosity>0) cout <<"DCC Error Type 3"<<endl;
    mapDCCproblem(dcc_);
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
      if (fVerbosity>0) cout <<"DCC Error Type 4"<<endl;
      mapDCCproblem(dcc_);
    }
  }
  
  unsigned char HTRErrorList=0; 
  for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
    HTRErrorList=dccHeader->getSpigotErrorBits(j);    
  }

  // These will be used in FED-vs-spigot 2D Histograms
  const int fed3offset = 1 + (4*dcc_); //3 bins, plus one of margin, each DCC
  const int fed2offset = 1 + (3*dcc_); //2 bins, plus one of margin, each DCC
  if (TTS_state & 0x8) /*RDY*/ 
    ;
  if (TTS_state & 0x2) /*SYN*/ 
    {
      if (fVerbosity>0) cout <<"TTS_state Error:sync"<<endl;
      mapDCCproblem(dcc_);          // DCC lost data
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
	mapHTRproblem(dcc_, spigot); 
      }
      if (((WholeErrorList>>2)&0x01)!=0)  {//Truncated data coming into LRB
	LRBDataCorruptionIndicators_[fed3offset+2][spg3offset+2]++;
	mapHTRproblem(dcc_, spigot); 
      }
      if (((WholeErrorList>>3)&0x01)!=0)  {//FIFO Overflow
	LRBDataCorruptionIndicators_[fed3offset+1][spg3offset+1]++;
	mapHTRproblem(dcc_, spigot); 
      }
      if (((WholeErrorList>>4)&0x01)!=0)  {//ID (EvN Mismatch), htr payload metadeta
	LRBDataCorruptionIndicators_[fed3offset+2][spg3offset+1]++;
	mapHTRproblem(dcc_, spigot); 
      }
      if (((WholeErrorList>>5)&0x01)!=0)  {//STatus: hdr/data/trlr error
	LRBDataCorruptionIndicators_[fed3offset+1][spg3offset+0]++;
	mapHTRproblem(dcc_, spigot); 
      }
      if (((WholeErrorList>>6)&0x01)!=0)  {//ODD 16-bit word count from HTR
	LRBDataCorruptionIndicators_[fed3offset+2][spg3offset+0]++;
	mapHTRproblem(dcc_, spigot); 
      }
    }
    if (!dccHeader->getSpigotPresent((unsigned int) spigot)){
      LRBDataCorruptionIndicators_[fed3offset+0][spg3offset+2]++;  //Enabled, but data not present!
      if (fVerbosity>0) cout <<"HTR Problem: Spigot Not Present"<<endl;
      mapHTRproblem(dcc_, spigot);
    } else {
      //?////I got the wrong sign on getBxMismatchWithDCC; 
      //?////It's a match, not a mismatch, when true. I'm sorry. 
      //?//if (!dccHeader->getBxMismatchWithDCC((unsigned int) spigot)) ;
      //?//if (!dccHeader->getSpigotValid((unsigned int) spigot)      ) ;//actually "EvN match" 
      if ( dccHeader->getSpigotDataTruncated((unsigned int) spigot)) {
     	LRBDataCorruptionIndicators_[fed3offset-1][spg3offset+0]++;  // EventBuilder truncated babbling LRB
	if (fVerbosity>0) cout <<"HTR Problem: Spigot Data Truncated"<<endl;
	mapHTRproblem(dcc_, spigot);
      }
      if ( dccHeader->getSpigotCRCError((unsigned int) spigot)) {
	LRBDataCorruptionIndicators_[fed3offset+0][spg3offset+0]++; 
	//Already mapped any HTR problem with this one.
      }
    } //else spigot marked "Present"
    if (dccHeader->getSpigotDataLength(spigot) <(unsigned long)4) {
      LRBDataCorruptionIndicators_[fed3offset+0][spg3offset+1]++;  //Lost HTR Data for sure.
      if (fVerbosity>0) cout <<"HTR Problem: Spigot Data Length too small"<<endl;
      mapHTRproblem(dcc_, spigot);
    }    
  }

  //Fake a problem with each DCC a unique number of times
  //if ((dcc_+1)>= ievt_)
  //  mapDCCproblem(dcc_); 

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
      if (fVerbosity>0) cout <<"HTR Problem: HTR check fails"<<endl;
      mapHTRproblem(dcc_,spigot);
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
      if (fVerbosity>0) cout <<"HTR Problem: NTP+NDAQ size consistency check fails"<<endl;
      mapHTRproblem(dcc_,spigot);
      //incompatible Sizes declared. Skip it.
      continue; }
    bool EE = ((dccHeader->getSpigotErrorBits(spigot) >> 2) & 0x01);
    if (EE) { 
      if (HTRwdcount != 8) {	//incompatible Sizes declared. Skip it.
	++HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+1];
	if (fVerbosity>0) cout <<"HTR Problem: HTRwdcount !=8"<<endl;	
	mapHTRproblem(dcc_,spigot);
      }
      DataFlowInd_[fed2offset+0][spg3offset+0]++;
      continue;}
    else{ //For non-EE, both CompactMode and !CompactMode
      bool CM = (htr.getExtHdr7() >> 14)&0x0001;
      if (( CM && ( (HTRwdcount-NDAQ-NTP) != 12) )
	  ||                                
	  (!CM && ( (HTRwdcount-NDAQ-NTP) != 20) )  ) {	//incompatible Sizes declared. Skip it.
	++HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+1];
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
      if (fVerbosity>0) cout <<"Orbit or BCN  HTR/DCC mismatch"<<endl;
      mapDCCproblem(dcc_);
    }
    if ( (htrEvtN == dccEvtNum) != 
	 dccHeader->getSpigotValid(spigot) ) {
      meDCCEventFormatError_->Fill(dccid,5);
      if (fVerbosity>0) cout <<"DCC invalid spigot"<<endl;
      mapDCCproblem(dcc_);
    }
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    if (prtlvl_ > 0) HTRPrint(htr,prtlvl_);
    unsigned int htrFWVer = htr.getFirmwareRevision() & 0xFF;
    meHTRFWVersion_->Fill(cratenum,htrFWVer);  

    ///check that all HTRs have the same L1A number.
    int EvtNdiff = htrEvtN - dccEvtNum;
    if (EvtNdiff!=0) {
      meEvtNumberSynch_->Fill(dcc_,spigot);
      meEvtNCheck_->Fill(EvtNdiff);
      if (prtlvl_ == 1)cout << "++++ Evt # out of sync, ref, this HTR: "<< dccEvtNum << "  "<<htrEvtN <<endl;
    }

    ///check that all HTRs have the same BCN
    int BCNdiff = htrBCN-dccBCN;
    if (BCNdiff!=0) {
      meBCNSynch_->Fill(dcc_,spigot);
      meBCNCheck_->Fill(BCNdiff);
      if (prtlvl_==1)cout << "++++ BCN # out of sync, ref, this HTR: "<< dccBCN << "  "<<htrBCN <<endl;
    }

    ///check that all HTRs have the same OrN
    int OrNdiff = htrOrN-dccOrN;
    if (OrNdiff!=0) {
      meOrNSynch_->Fill(dcc_,spigot);
      meOrNCheck_->Fill(OrNdiff);
      meBCNwhenOrNDiff_->Fill(htrBCN); // Are there special BCN where OrN mismatched occur? Let's see.
      if (prtlvl_==1)cout << "++++ OrN # out of sync, ref, this HTR: "<< dccOrN << "  "<<htrOrN <<endl;
    }

    bool htrUnSuppressed=(HTRraw[6]>>15 & 0x0001);
    if (htrUnSuppressed) {
      UScount[dcc_][spigot]++;
      int here=1+(HcalDCCHeader::SPIGOT_COUNT*(dcc_))+spigot;
      meUSFractSpigs_->setBinContent(here,
				     ((double)UScount[dcc_][spigot])/(double)ievt_);}

    //Fake a problem with each HTR a unique number of times.
    //if ( (spigot+1) >= ievt_ ) 
    //mapHTRproblem(dcc_,spigot); 

    //Fake a problem with each real calorimeter cell a unique number of times.
    //for (int htrchan=1; htrchan<=HTRCHANMAX; htrchan++) {
    //  //  if (htrchan>ievt_)
    //  mapChannproblem(dcc_,spigot,htrchan); }

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
      if (fVerbosity>0) cout <<"DCC spigot summary error or HTR error word"<<endl;
      mapDCCproblem(dcc_);                  //What other problems may lurk? Spooky.
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
	  case (14): //CT
	    HalfHTRDataCorruptionIndicators_[fed3offset+0][spg3offset+2]++;
	    if (fVerbosity>0) cout <<"HTR Problem: Case 14"<<endl;
	    mapHTRproblem(dcc_,spigot); break;
	  case (13): //HM
	    HalfHTRDataCorruptionIndicators_[fed3offset+0][spg3offset+1]++;
	    if (fVerbosity>0) cout <<"HTR Problem: Case 13"<<endl;
	    mapHTRproblem(dcc_,spigot); break;
	  case (12): //TM
	    HalfHTRDataCorruptionIndicators_[fed3offset+0][spg3offset+0]++;
	    if (fVerbosity>0) cout <<"HTR Problem: Case 12"<<endl;
	    mapHTRproblem(dcc_,spigot); break;
	  case ( 8): //BE
	    HalfHTRDataCorruptionIndicators_[fed3offset+1][spg3offset+2]++;
	    if (fVerbosity>0) cout <<"HTR Problem: Case 8"<<endl;
	    mapHTRproblem(dcc_,spigot); break;
	  case (15): //b15
	    HalfHTRDataCorruptionIndicators_[fed3offset+1][spg3offset+1]++;
	    mapHTRproblem(dcc_,spigot); break;
	  case ( 7): //CK
	    HalfHTRDataCorruptionIndicators_[fed3offset+1][spg3offset+0]++;
	    if (fVerbosity>0) cout <<"HTR Problem: Case 7"<<endl;
	    mapHTRproblem(dcc_,spigot); break;
	  case ( 5): //LW
	    HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+2]++;
	    //Sometimes set spuriously at startup, per-fiber, .: Leniency: 8
	    if (HalfHTRDataCorruptionIndicators_[fed3offset+2][spg3offset+2] > 8) { 
	      if (fVerbosity>0) cout <<"HTR Problem: Case 5"<<endl;
	      mapHTRproblem(dcc_,spigot); break; 
	    }
	  case ( 3): //L1 (previous L1A violated trigger rules)
	    DataFlowInd_[fed2offset+1][spg3offset+0]++; break;
	  case ( 1): //BZ
	    DataFlowInd_[fed2offset+0][spg3offset+1]++; break;
	  case ( 0): //OW
	    DataFlowInd_[fed2offset+0][spg3offset+2]++;
	  default: break;
	  }
	  meErrWdCrate_->Fill(cratenum,i);
	  if      (cratenum == 0) meCrate0HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 1) meCrate1HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 2) meCrate2HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 3) meCrate3HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 4) meCrate4HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 5) meCrate5HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 6) meCrate6HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 7) meCrate7HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum == 9) meCrate9HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==10)meCrate10HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==11)meCrate11HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==12)meCrate12HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==13)meCrate13HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==14)meCrate14HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==15)meCrate15HTRErr_ -> Fill(slotnum,i);
	  else if (cratenum ==17)meCrate17HTRErr_ -> Fill(slotnum,i);
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
    
    int lastcapid=-1;
    int samplecounter=-1;
    int htrchan=-1; // Valid: [1,24]
    int chn2offset=0; 
    int NTS = htr.getNDD(); //number time slices, in precision channels
    ChannSumm_DataIntegrityCheck_  [fed2offset-1][spg2offset+0]=NTS;//For normalization by client
    // Run over DAQ words for this spigot
    for (qie_work=qie_begin; qie_work!=qie_end; qie_work++) {
      if (qie_work->raw()==0xFFFF)  // filler word
	continue;
      //Beginning a channel's samples?
      if (( 1 + ( 3* (qie_work->fiber()-1) ) + qie_work->fiberChan() )  != htrchan) { //new channel starting
	// A fiber [1..8] carries three fiber channels, each is [0..2]. Make htrchan [1..24]
	htrchan= (3* (qie_work->fiber()-1) ) + qie_work->fiberChan(); 
	chn2offset = (htrchan*3)+1;
	++ChannSumm_DataIntegrityCheck_  [fed2offset-1][spg2offset-1];//tally
	++Chann_DataIntegrityCheck_[dcc_][chn2offset-1][spg2offset-1];//tally
	if (samplecounter !=-1) { //Wrap up the previous channel if there is one
	  //Check the previous digi for number of timeslices
	  if ((samplecounter != NTS) &&
	      (samplecounter != 1)             ) 
	    { //Wrong DigiSize
	      ++ChannSumm_DataIntegrityCheck_  [fed2offset+0][spg2offset+0];
	      ++Chann_DataIntegrityCheck_[dcc_][chn2offset+0][spg2offset+0];
	      if (fVerbosity) cout <<"mapChannelProblem:  Wrong Digi Size"<<endl;
	      mapChannproblem(dcc_,spigot,htrchan); 
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
	  if (fVerbosity) cout <<"mapChannelProblem:  Wrong Cap ID"<<endl;
	  mapChannproblem(dcc_,spigot,htrchan); }
	lastcapid=qie_work->capid();
	samplecounter++;}
      //For every sample, whether the first of the channel or not, !DV, Er
      if (!(qie_work->dv())){
	++ChannSumm_DataIntegrityCheck_  [fed2offset+0][spg2offset+1];
	++Chann_DataIntegrityCheck_[dcc_][chn2offset+0][spg2offset+1]; }
      if (qie_work->er()) {      // FEE - Front End Error
	++ChannSumm_DataIntegrityCheck_  [fed2offset+1][spg2offset+1];
	++Chann_DataIntegrityCheck_[dcc_][chn2offset+1][spg2offset+1]; 
	if (fVerbosity) cout <<"mapChannelProblem:  FE Error"<<endl;	
	mapChannproblem(dcc_,spigot,htrchan); }
    } // for (qie_work = qie_begin;...)  end loop over all timesamples in this spigot
    //Wrap up the last channel
    //Check the last digi for number of timeslices
    if ((samplecounter != NTS) &&
	(samplecounter != 1)            &&
	(samplecounter !=-1)             ) { //Wrong DigiSize (unexpected num. timesamples)
      ++ChannSumm_DataIntegrityCheck_  [fed2offset+0][spg2offset+0];
      ++Chann_DataIntegrityCheck_[dcc_][chn2offset+0][spg2offset+0];
      if (fVerbosity) cout <<"mapChannelProblem:  Wrong Digi Size (last digi)"<<endl;
      mapChannproblem(dcc_,spigot,htrchan); 
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
} // loop over DCCs void HcalDataFormatMonitor::unpack(

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
  for (int thfed=0; thfed<NUMDCCS; thfed++) {
    sprintf(label, "%03d", thfed+700);
    me_ptr->setBinLabel((2+(thfed*xbins)), label, 1); //margin of 1 at low value
  }
}

// Public function so HcalMonitorModule can slip in a 
// logical map digest or two. 
void HcalDataFormatMonitor::stashHDI(int thehash, HcalDetId thehcaldetid) {
  //Let's not allow indexing off the array...
  if ((thehash<0)||(thehash>(NUMDCCS*NUMSPIGS*HTRCHANMAX)))return;
  //...but still do the job requested.
  hashedHcalDetId_[thehash] = thehcaldetid;
}

void HcalDataFormatMonitor::mapDCCproblem(int dcc) {
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
    if (myeta>=0 && myeta<85 &&
	(myphi-1)>=0 && (myphi-1)<72 &&
	(mydepth-1)>=0 && (mydepth-1)<4)
      problemfound[myeta][myphi-1][mydepth-1] = true;
    if (fVerbosity>0)
      if (problemfound[myeta][myphi-1][mydepth-1])
	cout<<" mapDCCproblem error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<endl;
  }
}
void HcalDataFormatMonitor::mapHTRproblem(int dcc, int spigot) {
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
    if (myeta>=0 && myeta<85 &&
	(myphi-1)>=0 && (myphi-1)<72 &&
	(mydepth-1)>=0 && (mydepth-1)<4)
      problemfound[myeta][myphi-1][mydepth-1] = true;
    if (fVerbosity>0)
      if (problemfound[myeta][myphi-1][mydepth-1])
	cout<<" mapHTRproblem error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<endl;
    
  }
}   // void HcalDataFormatMonitor::mapHTRproblem(...)

void HcalDataFormatMonitor::mapChannproblem(int dcc, int spigot, int htrchan) {
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
  if (myeta>=0 && myeta<85 &&
      (myphi-1)>=0 && (myphi-1)<72 &&
      (mydepth-1)>=0 && (mydepth-1)<4)
    problemfound[myeta][myphi-1][mydepth-1] = true;

  if (fVerbosity>0)
    if (problemfound[myeta][myphi-1][mydepth-1])
      cout<<" mapChannproblem error! "<<HDI.subdet()<<"("<<HDI.ieta()<<", "<<HDI.iphi()<<", "<<HDI.depth()<<")"<<endl;
}   // void HcalDataFormatMonitor::mapChannproblem(...)

void HcalDataFormatMonitor::endLuminosityBlock(void) {
  if (LBprocessed_==true) return;  // LB already processed
  UpdateMEs();
  LBprocessed_=true; 
  return;
}

void HcalDataFormatMonitor::UpdateMEs (void ) {
  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      if (LRBDataCorruptionIndicators_  [x][y])
	meLRBDataCorruptionIndicators_->setBinContent(x+1,y+1,LRBDataCorruptionIndicators_[x][y]);
  	 
  for (int x=0; x<THREE_FED; x++)
    for (int y=0; y<THREE_SPG; y++)
      if (HalfHTRDataCorruptionIndicators_  [x][y])
	meHalfHTRDataCorruptionIndicators_->setBinContent(x+1,y+1,HalfHTRDataCorruptionIndicators_[x][y]);
  	 
  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<TWO__SPGT; y++)
      if (ChannSumm_DataIntegrityCheck_[x][y])
	meChannSumm_DataIntegrityCheck_->setBinContent(x+1,y+1,ChannSumm_DataIntegrityCheck_[x][y]);

  for (int f=0; f<NUMDCCS; f++)
    for (int x=0; x<TWO_CHANN; x++)
      for (int y=0; y<TWO__SPGT; y++)      
	if (Chann_DataIntegrityCheck_[f][x][y])
	  meChann_DataIntegrityCheck_[f]->setBinContent(x+1,y+1,Chann_DataIntegrityCheck_ [f][x][y]);

  for (int x=0; x<TWO___FED; x++)
    for (int y=0; y<THREE_SPG; y++)      
      if (DataFlowInd_[x][y])
	meDataFlowInd_->setBinContent(x+1,y+1,DataFlowInd_[x][y]);

  uint64_t probcnt=0;

  int etabins=0;
  int phibins=0;
  int filleta=-9999;
  
  ProblemCells->Reset(); // clear old values so that we can use "Fill" without problems
  ProblemCells->setBinContent(0,0,ievt_);
  for (int depth=0;depth<4;++depth)
    {
      etabins=ProblemCellsByDepth.depth[depth]->getNbinsX();
      phibins=ProblemCellsByDepth.depth[depth]->getNbinsY();
      ProblemCellsByDepth.depth[depth]->Reset(); // clear depth histograms
      ProblemCellsByDepth.depth[depth]->setBinContent(0,0,ievt_); // set underflow bin to event count
      for (int eta=0;eta<etabins;++eta)
	{
	  for (int phi=0;phi<phibins;++phi)
	    {
	      probcnt=((uint64_t) problemcount[eta][phi][depth] );
	      if (probcnt==0) continue;
	      filleta=CalcIeta(eta,depth+1); // calculate ieta from eta counter
	      // Offset true ieta for HF plotting
	      if (isHF(eta,depth+1)) 
		filleta<0 ? filleta-- : filleta++;
	      cout <<"probcnt = "<<probcnt<<"  ieta = "<<filleta<<"  iphi = "<<phi+1<<"  depth = "<<depth+1<<endl;
	      ProblemCellsByDepth.depth[depth]->Fill(filleta,phi+1,probcnt);
		ProblemCells->Fill(filleta,phi+1,probcnt); 
	    }
	}
    }
  // Make sure problem rate (summed over depths) doesn't exceed 100%
  etabins=ProblemCells->getNbinsX();
  phibins=ProblemCells->getNbinsY();
  for (int eta=0;eta<etabins;++eta)
    {
      for (int phi=0;phi<phibins;++phi)
  	{
	  if (ProblemCells->getBinContent(eta+1,phi+1)>ievt_)
  	    ProblemCells->setBinContent(eta+1,phi+1,ievt_);
  	}
    }
  
  FillUnphysicalHEHFBins(ProblemCells);
  FillUnphysicalHEHFBins(ProblemCellsByDepth);
} //UpdateMEs
