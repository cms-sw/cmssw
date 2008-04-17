#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalDataFormatMonitor::HcalDataFormatMonitor() {}

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
  firstFED_ = FEDNumbering::getHcalFEDIds().first;
  for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++) {
    if(fVerbosity) cout << "[DFMon:]Pushback for fedUnpackList_: " << i <<endl;
    fedUnpackList_.push_back(i);
  }

  prtlvl_ = ps.getUntrackedParameter<int>("dfPrtLvl");

  if ( m_dbe ) {
    m_dbe->setCurrentFolder(baseFolder_);
    
    meEVT_ = m_dbe->bookInt("Data Format Task Event Number");
    meEVT_->Fill(ievt_);

    char* type = "DCC Ev Fragment Size Distribution";
    meFEDRawDataSizes_=m_dbe->book1D(type,type,200000,-0.5,200000.5);
    meFEDRawDataSizes_->setAxisTitle("# of Event Fragments",1);
    meFEDRawDataSizes_->setAxisTitle("# of bytes",2);

    type = "Spigot Format Errors";
    meSpigotFormatErrors_=  m_dbe->book1D(type,type,500,-1,999);
    meSpigotFormatErrors_->setAxisTitle("# of Errors",1);
    meSpigotFormatErrors_->setAxisTitle("# of Events",2);
    type = "Num Bad Quality Digis -DV bit-Err bit-Cap Rotation";
    meBadQualityDigis_=  m_dbe->book1D(type,type,4550,-1,9099);
    meBadQualityDigis_->setAxisTitle("# of Bad Digis",1);
    meBadQualityDigis_->setAxisTitle("# of Events",2);
    type = "Num Unmapped Digis";
    meUnmappedDigis_=  m_dbe->book1D(type,type,4550,-1,9099);
    meUnmappedDigis_->setAxisTitle("# of Unmapped Digis",1);
    meUnmappedDigis_->setAxisTitle("# of Events",2);
    type = "Num Unmapped Trigger Primitive Digis";
    meUnmappedTPDigis_=  m_dbe->book1D(type,type,4550,-1,9099);
    meUnmappedTPDigis_->setAxisTitle("# of Unmapped Trigger Primitive Digis",1);
    meUnmappedTPDigis_->setAxisTitle("# of Events",2);
    type = "FED Error Map from Unpacker Report";
    meFEDerrorMap_ = m_dbe->book1D(type,type,33,699.5,732.5);
    meFEDerrorMap_->setAxisTitle("Dcc Id",1);
    meFEDerrorMap_->setAxisTitle("# of Errors",2);
    type = "EvtNum Not Consistent Within Spigots of a DCC";
    meEvtNumberSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    meEvtNumberSynch_->setAxisTitle("Slot #",1);
    meEvtNumberSynch_->setAxisTitle("Crate #",2);
    type = "BCN Not Consistent Within Spigots of DCC";
    meBCNSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    meBCNSynch_->setAxisTitle("Slot #",1);
    meBCNSynch_->setAxisTitle("Crate #",2);
    type = "BCN from HTRs";
    meBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
    meBCN_->setAxisTitle("BCN",1);
    meBCN_->setAxisTitle("# of Entries",2);
    type = "BCN Differences Among Spigots of a DCC";
    meBCNCheck_ = m_dbe->book1D(type,type,501,-250.5,250.5);
    meBCNCheck_->setAxisTitle("htr BCN - dcc BCN",1);
    type = "EvN Differences Among Spigots of a DCC";
    meEvtNCheck_ = m_dbe->book1D(type,type,601,-300.5,300.5);
    meEvtNCheck_->setAxisTitle("htr Evt # - dcc Evt #",1);
    type = "BCN of Fiber Orbit Message";
    meFibBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
    meFibBCN_->setAxisTitle("BCN of Fib Orb Msg",1);
    // Firmware version
    type = "HTR Firmware Version";
    meFWVersion_ = m_dbe->book2D(type,type ,256,-0.5,255.5,18,-0.5,17.5);
    meFWVersion_->setAxisTitle("HTR Firmware Version",1);
    meFWVersion_->setAxisTitle("Crate #",2);
    // Examine conditions of the DCC Event Fragment
    type = "Number of Event Fragments by FED ID";
    meFEDId_=m_dbe->book1D(type, type, 35, 729.5, 733.5);
    meFEDId_->setAxisTitle("All possible values of HCAL FED ID",1);

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

    type = "DCC Status Flags (Nonzero Error Counters)";
    meDCCStatusFlags_ = m_dbe->book2D(type,type,32,699.5,731.5,9,0.5,9.5);
    meDCCStatusFlags_->setAxisTitle("HCAL FED ID", 1);      
    meDCCStatusFlags_->setBinLabel(1, "Saw OFW", 2);
    meDCCStatusFlags_->setBinLabel(2, "Saw BSY", 2);
    meDCCStatusFlags_->setBinLabel(3, "Saw SYN", 2);
    meDCCStatusFlags_->setBinLabel(4, "MxMx_L1AEvN", 2);
    meDCCStatusFlags_->setBinLabel(5, "MxMx_L1ABcN", 2);
    meDCCStatusFlags_->setBinLabel(6, "MxMx_CT-EvN", 2);
    meDCCStatusFlags_->setBinLabel(7, "MxMx_CT-BCN", 2);
    meDCCStatusFlags_->setBinLabel(8, "BC0 Spacing", 2);
    meDCCStatusFlags_->setBinLabel(9, "TTCSingErr", 2);
    meDCCStatusFlags_->setBinLabel(10, "TTCDoubErr", 2);

    type = "DCC Error and Warning";
    meDCCErrorAndWarnConditions_ = m_dbe->book2D(type,type,32,699.5,731.5, 25,0.5,24.5);
    meDCCErrorAndWarnConditions_->setAxisTitle("HCAL FED ID", 1);      
    meDCCErrorAndWarnConditions_->setBinLabel(1, "Err Spgt 15", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(2, "Err Spgt 14", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(3, "Err Spgt 13", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(4, "Err Spgt 12", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(5, "Err Spgt 11", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(6, "Err Spgt 10", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(7, "Err Spgt 9", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(8, "Err Spgt 8", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(9, "Err Spgt 7", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(10, "Err Spgt 6", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(11, "Err Spgt 5", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(12, "Err Spgt 4", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(13, "Err Spgt 3", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(14, "Err Spgt 2", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(15, "Err Spgt 1 (top)", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(16, "TTS_OFW", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(17, "TTS_BSY", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(18, "TTS_SYN", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(19, "L1A_EvN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(20, "L1A_BcN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(21, "CT_EvN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(22, "CT_BcN Mis", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(23, "OrbitLenErr", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(24, "TTC_SingErr", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(25, "TTC_DoubErr", 2);

    //type = "DCC View of Spigot Conditions";
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

    int maxbits = 16;//Look at all bits
    type = "HTR Error Word by Crate";
    meErrWdCrate_ = m_dbe->book2D(type,type,18,-0.5,17.5,maxbits,-0.5,maxbits-0.5);
    meErrWdCrate_ -> setAxisTitle("Crate #",1);
    meErrWdCrate_ -> setBinLabel(1,"Overflow Warn",2);
    meErrWdCrate_ -> setBinLabel(2,"Buffer Busy",2);
    meErrWdCrate_ -> setBinLabel(3,"Empty Event",2);
    meErrWdCrate_ -> setBinLabel(4,"Reject L1A",2);
    meErrWdCrate_ -> setBinLabel(5,"Latency Err",2);
    meErrWdCrate_ -> setBinLabel(6,"Latency Warn",2);
    meErrWdCrate_ -> setBinLabel(7,"OptDat Err",2);
    meErrWdCrate_ -> setBinLabel(8,"Clock Err",2);
    meErrWdCrate_ -> setBinLabel(9,"Bunch Err",2);
    meErrWdCrate_ -> setBinLabel(13,"Test Mode",2);
    meErrWdCrate_ -> setBinLabel(14,"Histo Mode",2);
    meErrWdCrate_ -> setBinLabel(15,"Calib Trig",2);
    meErrWdCrate_ -> setBinLabel(16,"Bit15 Err",2);

    type = "HTR Error Word - Crate 0";
    meCrate0HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate0HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate0HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 1";
    meCrate1HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate1HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate1HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 2";
    meCrate2HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate2HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate2HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 3";
    meCrate3HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate3HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate3HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 4";
    meCrate4HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate4HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate4HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 5";
    meCrate5HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate5HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate5HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 6";
    meCrate6HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate6HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate6HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 7";
    meCrate7HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate7HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate7HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 8";
    meCrate8HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate8HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate8HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 9";
    meCrate9HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate9HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate9HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 10";
    meCrate10HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate10HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate10HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 11";
    meCrate11HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate11HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate11HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 12";
    meCrate12HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate12HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate12HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 13";
    meCrate13HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate13HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate13HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 14";
    meCrate14HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate14HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate14HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 15";
    meCrate15HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate15HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate15HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 16";
    meCrate16HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate16HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate16HTRErr_ ->setAxisTitle("Crate #",2);
    type = "HTR Error Word - Crate 17";
    meCrate17HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    meCrate17HTRErr_ ->setAxisTitle("Slot #",1);
    meCrate17HTRErr_ ->setAxisTitle("Crate #",2);
    
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

    type = "HBHE Data Format Error Word";
    DCC_ErrWd_HBHE =  m_dbe->book1D(type,type,16,-0.5,15.5);
    DCC_ErrWd_HBHE -> setBinLabel(1,"Overflow Warn",1);
    DCC_ErrWd_HBHE -> setBinLabel(2,"Buffer Busy",1);
    DCC_ErrWd_HBHE -> setBinLabel(3,"Empty Event",1);
    DCC_ErrWd_HBHE -> setBinLabel(4,"Reject L1A",1);
    DCC_ErrWd_HBHE -> setBinLabel(5,"Latency Err",1);
    DCC_ErrWd_HBHE -> setBinLabel(6,"Latency Warn",1);
    DCC_ErrWd_HBHE -> setBinLabel(7,"OptDat Err",1);
    DCC_ErrWd_HBHE -> setBinLabel(8,"Clock Err",1);
    DCC_ErrWd_HBHE -> setBinLabel(9,"Bunch Err",1);
    DCC_ErrWd_HBHE -> setBinLabel(13,"Test Mode",1);
    DCC_ErrWd_HBHE -> setBinLabel(14,"Histo Mode",1);
    DCC_ErrWd_HBHE -> setBinLabel(15,"Calib Trig",1);
    DCC_ErrWd_HBHE -> setBinLabel(16,"Bit15 Err",1);

    type = "HF Data Format Error Word";
    DCC_ErrWd_HF =  m_dbe->book1D(type,type,16,-0.5,15.5);
    DCC_ErrWd_HF -> setBinLabel(1,"Overflow Warn",1);
    DCC_ErrWd_HF -> setBinLabel(2,"Buffer Busy",1);
    DCC_ErrWd_HF -> setBinLabel(3,"Empty Event",1);
    DCC_ErrWd_HF -> setBinLabel(4,"Reject L1A",1);
    DCC_ErrWd_HF -> setBinLabel(5,"Latency Err",1);
    DCC_ErrWd_HF -> setBinLabel(6,"Latency Warn",1);
    DCC_ErrWd_HF -> setBinLabel(7,"OptDat Err",1);
    DCC_ErrWd_HF -> setBinLabel(8,"Clock Err",1);
    DCC_ErrWd_HF -> setBinLabel(9,"Bunch Err",1);
    DCC_ErrWd_HF -> setBinLabel(13,"Test Mode",1);
    DCC_ErrWd_HF -> setBinLabel(14,"Histo Mode",1);
    DCC_ErrWd_HF -> setBinLabel(15,"Calib Trig",1);
    DCC_ErrWd_HF -> setBinLabel(16,"Bit15 Err",1);

    type = "HO Data Format Error Word";
    DCC_ErrWd_HO = m_dbe->book1D(type,type,16,-0.5,15.5);
    DCC_ErrWd_HO -> setBinLabel(1,"Overflow Warn",1);
    DCC_ErrWd_HO -> setBinLabel(2,"Buffer Busy",1);
    DCC_ErrWd_HO -> setBinLabel(3,"Empty Event",1);
    DCC_ErrWd_HO -> setBinLabel(4,"Reject L1A",1);
    DCC_ErrWd_HO -> setBinLabel(5,"Latency Err",1);
    DCC_ErrWd_HO -> setBinLabel(6,"Latency Warn",1);
    DCC_ErrWd_HO -> setBinLabel(7,"OptDat Err",1);
    DCC_ErrWd_HO -> setBinLabel(8,"Clock Err",1);
    DCC_ErrWd_HO -> setBinLabel(9,"Bunch Err",1);
    DCC_ErrWd_HO -> setBinLabel(13,"Test Mode",1);
    DCC_ErrWd_HO -> setBinLabel(14,"Histo Mode",1);
    DCC_ErrWd_HO -> setBinLabel(15,"Calib Trig",1);
    DCC_ErrWd_HO -> setBinLabel(16,"Bit15 Err",1);

   }

   return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw, 
					 const HcalUnpackerReport& report, 
					 const HcalElectronicsMap& emap){
  
  if(!m_dbe) { 
    printf("HcalDataFormatMonitor::processEvent DQMStore not instantiated!!!\n");  
    return;
  }
  
  ievt_++;
  meEVT_->Fill(ievt_);
  
  meSpigotFormatErrors_->Fill(report.spigotFormatErrors());
  meBadQualityDigis_->Fill(report.badQualityDigis());
  meUnmappedDigis_->Fill(report.unmappedDigis());
  meUnmappedTPDigis_->Fill(report.unmappedTPDigis());

  lastEvtN_ = -1;
  lastBCN_ = -1;

  for (vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    if (fed.size()<12) continue; // Was 16. How do such tiny events even get here?
    unpack(fed,emap);
  }

  for(unsigned int i=0; i<report.getFedsError().size(); i++){
    const int m = report.getFedsError()[i];
    const FEDRawData& fed = rawraw.FEDData(m);
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(fed.data());
    if(!dccHeader) continue;
    int dccid=dccHeader->getSourceId();
    meFEDerrorMap_->Fill(dccid);
  }

   return;
}

void HcalDataFormatMonitor::unpack(const FEDRawData& raw, 
				   const HcalElectronicsMap& emap){
  
  // get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;

  // get the DCC trailer 
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  //DCC Event Fragment sizes distribution, in bytes.
  meFEDRawDataSizes_->Fill(raw.size());

  int dccid=dccHeader->getSourceId();
  if(fVerbosity) cout << "DCC " << dccid << endl;
  unsigned long dccEvtNum = dccHeader->getDCCEventNumber();
  int dccBCN = dccHeader->getBunchId();

  //There should never be HCAL DCCs reporting a fed id outside [700:731]
  meFEDId_->Fill(dccid);

  ////////// Histogram problems with the Common Data Format compliance;////////////
  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()) 
    meCDFErrorFound_->Fill(dccid, 1);
  /* 2 */ //Make sure a reference CDF Version value has been recorded for this dccid
  CDFvers_it = CDFversionNumber_list.find(dccid);
  if (CDFvers_it  == CDFversionNumber_list.end()) {
    CDFversionNumber_list.insert(pair<int,short>
				 (dccid,dccHeader->getCDFversionNumber() ) );
    CDFvers_it = CDFversionNumber_list.find(dccid);
  } // then check against it.
  if (dccHeader->getCDFversionNumber()!= CDFvers_it->second) 
    meCDFErrorFound_->Fill(dccid,2);
  /* 3 */ //Make sure a reference CDF EventType value has been recorded for this dccid
  CDFEvT_it = CDFEventType_list.find(dccid);
  if (CDFEvT_it  == CDFEventType_list.end()) {
    CDFEventType_list.insert(pair<int,short>
				 (dccid,dccHeader->getCDFEventType() ) );
    CDFEvT_it = CDFEventType_list.find(dccid);
  } // then check against it.
  if (dccHeader->getCDFEventType()!= CDFEvT_it->second) 
    meCDFErrorFound_->Fill(dccid,3);
  /* 4 */ //There should always be a '5' in CDF Header word 0, bits [63:60]
  if (dccHeader->BOEshouldBe5Always()!=5) 
    meCDFErrorFound_->Fill(dccid, 4);
  /* 5 */ //There should never be a third CDF Header word indicated.
  if (dccHeader->thereIsAThirdCDFHeaderWord())   
    meCDFErrorFound_->Fill(dccid, 5);
  /* 6 */ //Make sure a reference value of te Reserved Bits has been recorded for this dccid
  CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  if (CDFReservedBits_it  == CDFReservedBits_list.end()) {
    CDFReservedBits_list.insert(pair<int,short>
				 (dccid,dccHeader->getSlink64ReservedBits() ) );
    CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  } // then check against it.
  if ((int) dccHeader->getSlink64ReservedBits()!= CDFReservedBits_it->second)
    meCDFErrorFound_->Fill(dccid,6);
  /* 7 */ //There should always be 0x0 in CDF Header word 1, bits [63:60]
  if (dccHeader->BOEshouldBeZeroAlways() !=0)
    meCDFErrorFound_->Fill(dccid, 7);
  /* 8 */ //There should only be one trailer
  if (trailer.moreTrailers())
    meCDFErrorFound_->Fill(dccid, 8);
  //  if trailer.
  /* 9 */ //CDF Trailer [55:30] should be the # 64-bit words in the EvFragment
  if ((uint64_t) raw.size() != ( (uint64_t) trailer.lenght()*sizeof(uint64_t)) )  //The function name is a typo! Awesome.
    meCDFErrorFound_->Fill(dccid, 9);
  /*10 */ //There is a rudimentary sanity check built into the FEDTrailer class
  if (!trailer.check())
    meCDFErrorFound_->Fill(dccid, 10);
  

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
        //int SpigotPad = HcalDCCHeader::SPIGOT_COUNT;
  if (false) //(  ((uint64_t) dccHeader->getSpigotSummary(SpigotPad)  )
 	//        | ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+1)) 
	//        | ((uint64_t) dccHeader->getSpigotSummary(SpigotPad+2)) ) != 0)
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
    if (dccHeader->getSpigotErrorFlag(i))  meDCCErrorAndWarnConditions_->Fill(dccid, i);
  /* [16:25] */ //Histogram DCC Error and Warning Counters being nonzero
  if (false) /*dccHeader->SawTTS_OFW()        )*/  meDCCErrorAndWarnConditions_->Fill(dccid,16);
  if (false) /*dccHeader->SawTTS_BSY()        )*/  meDCCErrorAndWarnConditions_->Fill(dccid,17);
  if (false) /*dccHeader->SawTTS_SYN()        )*/  meDCCErrorAndWarnConditions_->Fill(dccid,18);
  if (false) /*dccHeader->SawL1A_EvN_MxMx()   )*/  meDCCErrorAndWarnConditions_->Fill(dccid,19);
  if (false) /*dccHeader->SawL1A_BcN_MxMx()   )*/  meDCCErrorAndWarnConditions_->Fill(dccid,20);
  if (false) /*dccHeader->SawCT_EvN_MxMx()    )*/  meDCCErrorAndWarnConditions_->Fill(dccid,21);
  if (false) /*dccHeader->SawCT_BcN_MxMx()    )*/  meDCCErrorAndWarnConditions_->Fill(dccid,22);
  if (false) /*dccHeader->SawOrbitLengthErr() )*/  meDCCErrorAndWarnConditions_->Fill(dccid,23);
  if (false) /*dccHeader->SawTTC_SingErr()    )*/  meDCCErrorAndWarnConditions_->Fill(dccid,24);
  if (false) /*dccHeader->SawTTC_DoubErr()    )*/  meDCCErrorAndWarnConditions_->Fill(dccid,25);


  ////////// Histogram Spigot Errors from the DCCs HTR Summaries;////////////
  /* [1:8] */ //Histogram HTR Error Bits in the DCC Headers
  bool FoundOne;
  unsigned char WholeErrorList=0; 
  for (int i=0; i<8; i++) {
    FoundOne=false;
    for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
      WholeErrorList=dccHeader->getSpigotErrorBits((unsigned int) j);
      if ((WholeErrorList>>i)&0x01) FoundOne=true;}
    if (FoundOne) meDCCSummariesOfHTRs_->Fill(dccid, i+1);
  }
  /* [9:16] */ //Histogram LRB Error Bits in the DCC Headers
  WholeErrorList=0; 
  for (int i=0; i<8; i++) {
    FoundOne=false;
    for(int j=0; j<HcalDCCHeader::SPIGOT_COUNT; j++) {
      WholeErrorList=dccHeader->getLRBErrorBits((unsigned int) j);
      if ((WholeErrorList>>i)&0x01) FoundOne=true;}
    if (FoundOne) meDCCSummariesOfHTRs_->Fill(dccid, i+9);
  }
  /* [17:20] */ //Histogram condition of Enabled Spigots without data Present
  bool FoundEnotP=false;
  bool FoundPnotB=false;
  bool FoundPnotV=false;
  bool FoundT=false;
  for(int j=1; j<=HcalDCCHeader::SPIGOT_COUNT; j++) {
    if (dccHeader->getSpigotEnabled((unsigned int) j) &&
	!dccHeader->getSpigotPresent((unsigned int) j)      ) FoundEnotP=true;
    //I got the wrong sign on getBxMismatchWithDCC; 
    //It's a match, not a mismatch, when true. I'm sorry. 
    if (dccHeader->getSpigotPresent((unsigned int) j) &&
	!dccHeader->getBxMismatchWithDCC((unsigned int) j)  ) FoundPnotB=true;
    if (dccHeader->getSpigotPresent((unsigned int) j) &&
	!dccHeader->getSpigotValid((unsigned int) j)        ) FoundPnotV=true;
    if (dccHeader->getSpigotDataTruncated((unsigned int) j) ) FoundT=true;
  }
  if (FoundEnotP)meDCCSummariesOfHTRs_->Fill(dccid,17);
  if (FoundPnotB)meDCCSummariesOfHTRs_->Fill(dccid,18);
  if (FoundPnotV)meDCCSummariesOfHTRs_->Fill(dccid,19);
  if (  FoundT  )meDCCSummariesOfHTRs_->Fill(dccid,20);

  // walk through the HTR data...
  HcalHTRData htr;  
  for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
    if (!dccHeader->getSpigotPresent(spigot)) continue;
    dccHeader->getSpigotData(spigot,htr);
    
    // check
    if (!htr.check()) continue;
    if (htr.isHistogramEvent()) continue;
    
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    if (prtlvl_ > 0) HTRPrint(htr,prtlvl_);

    unsigned int htrBCN = htr.getBunchNumber(); 
    meBCN_->Fill(htrBCN);
    unsigned int htrEvtN = htr.getL1ANumber();
    
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
    meFWVersion_->Fill(htrFWVer,cratenum);

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
   }

   return;
}

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




