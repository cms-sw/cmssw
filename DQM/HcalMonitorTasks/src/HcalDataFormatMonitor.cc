#include "DQM/HcalMonitorTasks/interface/HcalDataFormatMonitor.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

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
				  DaqMonitorBEInterface* dbe){
  HcalBaseMonitor::setup(ps,dbe);
  
  ievt_=0;
  baseFolder_ = rootFolder_+"DataFormatMonitor";

  if(fVerbosity) cout << "About to pushback fedUnpackList_" << endl;
  firstFED_ = FEDNumbering::getHcalFEDIds().first;
  for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++) {
    if(fVerbosity) cout << "Pushback for fedUnpackList_: " << i <<endl;
    fedUnpackList_.push_back(i);
  }

  prtlvl_ = ps.getUntrackedParameter<int>("dfPrtLvl");

  if ( m_dbe ) {
    m_dbe->setCurrentFolder(baseFolder_);
    
    meEVT_ = m_dbe->bookInt("Data Format Task Event Number");
    meEVT_->Fill(ievt_);
    
    char* type = "Spigot Format Errors";
    meSpigotFormatErrors_=  m_dbe->book1D(type,type,500,-1,999);
    meSpigotFormatErrors_->setAxisTitle("# of Errors",1);
    meSpigotFormatErrors_->setAxisTitle("# of Events",2);
    type = "Bad Quality Digis";
    meBadQualityDigis_=  m_dbe->book1D(type,type,4550,-1,9099);
    meBadQualityDigis_->setAxisTitle("# of Bad Digis",1);
    meBadQualityDigis_->setAxisTitle("# of Events",2);
    type = "Unmapped Digis";
    meUnmappedDigis_=  m_dbe->book1D(type,type,4550,-1,9099);
    meUnmappedDigis_->setAxisTitle("# of Unmapped Digis",1);
    meUnmappedDigis_->setAxisTitle("# of Events",2);
    type = "Unmapped Trigger Primitive Digis";
    meUnmappedTPDigis_=  m_dbe->book1D(type,type,4550,-1,9099);
    meUnmappedTPDigis_->setAxisTitle("# of Unmapped Trigger Primitive Digis",1);
    meUnmappedTPDigis_->setAxisTitle("# of Events",2);
    type = "FED Error Map";
    meFEDerrorMap_ = m_dbe->book1D(type,type,33,699.5,732.5);
      meFEDerrorMap_->setAxisTitle("Dcc Id",1);
      meFEDerrorMap_->setAxisTitle("# of Errors",2);
    type = "Evt Number Out-of-Synch";
    meEvtNumberSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
      meEvtNumberSynch_->setAxisTitle("Slot #",1);
      meEvtNumberSynch_->setAxisTitle("Crate #",2);
    type = "BCN Not Consistent";
    meBCNSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
       meBCNSynch_->setAxisTitle("Slot #",1);
       meBCNSynch_->setAxisTitle("Crate #",2);
    type = "BCN";
    meBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);
       meBCN_->setAxisTitle("BCN",1);
       meBCN_->setAxisTitle("# of Entries",2);
   
    type = "BCN Check";
    meBCNCheck_ = m_dbe->book1D(type,type,501,-250.5,250.5);
    meBCNCheck_->setAxisTitle("htr BCN - dcc BCN",1);
    type = "EvtN Check";
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
    type = "FED ID numbers. (Should be [700...731])";
    meFEDId_=m_dbe->book1D(type, type, 4095, -0.5, 4094.5);
    meFEDId_->setAxisTitle("All possible values of FED ID",1);

    type = "Event Fragments violating the Common Data Format";
    meCDFErrorFound_ = m_dbe->book2D(type,type,32,699.5,731.5,9,0.5,9.5);
    meCDFErrorFound_->setAxisTitle("HCAL FED ID", 1);
    meCDFErrorFound_->setBinLabel(1, "Missing Indicator for Second CDF Header", 2);
    meCDFErrorFound_->setBinLabel(2, "CDF Version Number Inconsistency", 2);
    meCDFErrorFound_->setBinLabel(3, "CDF Event Type Inconsistency", 2);
    meCDFErrorFound_->setBinLabel(4, "Beginning Of Event not '0x5' as Spec'ed", 2);
    meCDFErrorFound_->setBinLabel(5, "Erroneous Third CDF Header Indicated", 2);
    meCDFErrorFound_->setBinLabel(6, "Reserved Second Header Bits Inconsistent", 2);
    meCDFErrorFound_->setBinLabel(7, "Second BOE not '0x0;", 2);

    type = "DCC Event Format violation";
    meDCCEventFormatError_ = m_dbe->book2D(type,type,32,699.5,731.5,9,0.5,9.5);
    meDCCEventFormatError_->setAxisTitle("HCAL FED ID", 1);
    meDCCEventFormatError_->setBinLabel(1, "DCC Format Version Inconsistent", 2);
    meDCCEventFormatError_->setBinLabel(2, "DCC Reserved Bits Inconsistent", 2);
    meDCCEventFormatError_->setBinLabel(3, "DCC 'Zero Bits' Not Zero ", 2);
    meDCCEventFormatError_->setBinLabel(4, "DCC Header Bits [63:42] NonZero", 2);
    meDCCEventFormatError_->setBinLabel(5, "Nonvalid HTR Summaries 15, 16 ,17 NonZero", 2);
    meDCCEventFormatError_->setBinLabel(6, "Spigot Error Flag Miscalculated", 2);      
    meDCCEventFormatError_->setBinLabel(7, "LRB Truncation Bit MisCopied", 2);	       
    meDCCEventFormatError_->setBinLabel(8, "32-Bit Padding Word Needed But Absent", 2);
    meDCCEventFormatError_->setBinLabel(9, "Event Size Internally Misdescribed", 2);

    type = "DCC Error and Warning";
    meDCCErrorAndWarnConditions_ = m_dbe->book2D(type,type,32,699.5,731.5, 25,0.5,24.5);
    meDCCErrorAndWarnConditions_->setAxisTitle("HCAL FED ID", 1);
    meDCCErrorAndWarnConditions_->setBinLabel(1, "Error Spigot 15", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(2, "Error Spigot 14", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(3, "Error Spigot 13", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(4, "Error Spigot 12", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(5, "Error Spigot 11", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(6, "Error Spigot 10", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(7, "Error Spigot 9", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(8, "Error Spigot 8", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(9, "Error Spigot 7", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(10, "Error Spigot 6", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(11, "Error Spigot 5", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(12, "Error Spigot 4", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(13, "Error Spigot 3", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(14, "Error Spigot 2", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(15, "Error Spigot 1 (top)", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(16, "OverFlow Warning Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(17, "Busy (Ignore L1A) Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(18, "Synch Loss Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(19, "L1 EvN Mismatch Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(20, "L1 BcN Mismatch  Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(21, "Calibration EvN Mismatch Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(22, "Calibration BcN MismatchOccurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(23, "TTCrx Bx Error Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(24, "TTCrx Single-Bit Error Occurred", 2);
    meDCCErrorAndWarnConditions_->setBinLabel(25, "TTCrx Double-Bit Error Occurred", 2);

    type = "DCC View of Spigot Conditions";
    meDCCSummariesOfHTRs_ = m_dbe->book2D(type,type,32,699.5,731.5, 20,0.5,20.5);
    meDCCSummariesOfHTRs_->setAxisTitle("HCAL FED ID", 1);
    meDCCSummariesOfHTRs_->setBinLabel(1, "One or More HTR OverFlowWarnings Seen", 2);
    meDCCSummariesOfHTRs_->setBinLabel(2, "One or More Internal HTR Buffers Busy", 2);
    meDCCSummariesOfHTRs_->setBinLabel(3, "One or More Empty Events", 2);
    meDCCSummariesOfHTRs_->setBinLabel(4, "One or More L1A Rejected by a HTR", 2);
    meDCCSummariesOfHTRs_->setBinLabel(5, "One or More Latency Errors by a HTR", 2);
    meDCCSummariesOfHTRs_->setBinLabel(6, "One or More Latency Warnings by a HTR", 2);
    meDCCSummariesOfHTRs_->setBinLabel(7, "One or More Optical Data Errors by a HTR", 2);
    meDCCSummariesOfHTRs_->setBinLabel(8, "One or More Clocking Problems from a HTR", 2);
    meDCCSummariesOfHTRs_->setBinLabel(9, "One or More Corrected Link Errors from an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(10, "One or More Uncorrected Link Errors from an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(11, "One or More Block Size Overflows from an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(12, "One or More EvN Header/Trailer Mismatch from an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(13, "One or More FIFOs Empty when reading block by an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(14, "One or More Overflow (Data Truncated) from an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(15, "One or More Missing Header/trailer by an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(16, "One or More Odd 16-Bit Word Count by an LRB", 2);
    meDCCSummariesOfHTRs_->setBinLabel(17, "One or More Spigots Enabled without data Present", 2);
    meDCCSummariesOfHTRs_->setBinLabel(18, "One or More Spigots data Present without BcN Matching DCC", 2);
    meDCCSummariesOfHTRs_->setBinLabel(19, "One or More Spigots data Present but not valid", 2);
    meDCCSummariesOfHTRs_->setBinLabel(20, "One or More Spigots Truncated by LRB", 2);

    int maxbits = 16;//Look at all bits
    type = "HTR Error Word by Crate";
    meErrWdCrate_ = m_dbe->book2D(type,type,18,-0.5,17.5,maxbits,-0.5,maxbits-0.5);

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

    type = "HF Data Format Error Word";
    DCC_ErrWd_HF =  m_dbe->book1D(type,type,16,-0.5,15.5);

    type = "HO Data Format Error Word";
    DCC_ErrWd_HO = m_dbe->book1D(type,type,16,-0.5,15.5);

   }

   return;
}

void HcalDataFormatMonitor::processEvent(const FEDRawDataCollection& rawraw, 
					 const HcalUnpackerReport& report, 
					 const HcalElectronicsMap& emap){
  
  if(!m_dbe) { 
    printf("HcalDataFormatMonitor::processEvent DaqMonitorBEInterface not instantiated!!!\n");  
    return;
  }
  
  ievt_++;
  meEVT_->Fill(ievt_);
  

  meSpigotFormatErrors_->Fill(report.spigotFormatErrors());
  meBadQualityDigis_->Fill(report.badQualityDigis());

  if (report.badQualityDigis()> 3000) return;

  meUnmappedDigis_->Fill(report.unmappedDigis());
  meUnmappedTPDigis_->Fill(report.unmappedTPDigis());

  lastEvtN_ = -1;
  lastBCN_ = -1;

  for (vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    if (fed.size()<16) continue;
    unpack(fed,emap);
  }
  /*
  meSpigotFormatErrors_->Fill(report.spigotFormatErrors());
  meBadQualityDigis_->Fill(report.badQualityDigis());
  meUnmappedDigis_->Fill(report.unmappedDigis());
  meUnmappedTPDigis_->Fill(report.unmappedTPDigis());
  */
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
  
  // get the DataFormat header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;
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
  if (dccHeader->getSlink64ReservedBits()!= CDFReservedBits_it->second)
    meCDFErrorFound_->Fill(dccid,6);
  /* 7 */ //There should always be 0x0 in CDF Header word 1, bits [63:60]
  if (dccHeader->BOEshouldBeZeroAlways() !=0)
    meCDFErrorFound_->Fill(dccid, 7);
  
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
  /* 2 */ //Make sure a reference value of the DCC Reserved bits has been noted for this dcc.
  DCCRsvdBits_it = DCCRsvdBits_list.find(dccid);
  if (DCCRsvdBits_it == DCCRsvdBits_list.end()) {
    DCCRsvdBits_list.insert(pair<int,short>
				 (dccid,dccHeader->getDCCDataFormatVersion() ) );
    DCCRsvdBits_it = DCCRsvdBits_list.find(dccid);
  } // then check against it.
  if (dccHeader->getDCCHeaderSchmutz()!= DCCRsvdBits_it->second) 
    meDCCEventFormatError_->Fill(dccid,2);
  /* 3 */ //Check that there are zeros in the DCC Header above the HTR Status Bits
  if (dccHeader->getDCCHeaderZeros() !=0)
    meDCCEventFormatError_->Fill(dccid, 3);
  /* 4 */ //Check that there are zeros in the DCC Header above the 'DCC Counters Nonzero' Flags
  bool FoundOne = false;
  for(int i=9; i<31; i++) {
    if (dccHeader->isThisDCCErrorCounterNonZero((unsigned int) i)) FoundOne = true;}
  if (FoundOne)
    meDCCEventFormatError_->Fill(dccid, 4);
  /* 5 */ //Check that there are zeros in the DCC Header after the HTR Summaries.
  FoundOne = false;
  int NumberOfPaddingSpigots = 3;
  for(int i=HcalDCCHeader::SPIGOT_COUNT; 
      i<HcalDCCHeader::SPIGOT_COUNT+NumberOfPaddingSpigots; i++) {
    if (dccHeader->getSpigotDataLength(i) != 0) FoundOne=true;
    if (dccHeader->getSpigotEnabled((unsigned int)i) != 0) FoundOne=true;
    if (dccHeader->getSpigotPresent((unsigned int)i) != 0) FoundOne=true;
    if (dccHeader->getBxMismatchWithDCC((unsigned int)i) != 0) FoundOne=true;
    if (dccHeader->getSpigotValid((unsigned int) i) != 0) FoundOne=true;
    if (dccHeader->getSpigotDataTruncated((unsigned int) i) != 0) FoundOne=true;
    if ((int) dccHeader->getSpigotErrorBits((unsigned int) i) != 0) FoundOne=true;
    if ((int) dccHeader->getLRBErrorBits((unsigned int) i) != 0) FoundOne=true;
  }
  if (FoundOne) meDCCEventFormatError_->Fill(dccid, 5);
  /* 6 */ //Check that there are zeros following the HTR Payloads, if needed.
  int nHTR32BitWords=0;
  for(int i=0; i<HcalDCCHeader::SPIGOT_COUNT; i++) {
    nHTR32BitWords += dccHeader->getSpigotDataLength(i);  }
  if ( (( nHTR32BitWords % 2) == 1) && (true)) {
    meDCCEventFormatError_->Fill(dccid, 6); }
  
  ////////// Histogram Errors and Warnings from the DCC;////////////
  /* [1:15] */ //Histogram HTR Status Bits from the DCC Header
  for(int i=1; i<=HcalDCCHeader::SPIGOT_COUNT; i++)
    if (dccHeader->getSpigotErrorFlag(i))  meDCCErrorAndWarnConditions_->Fill(dccid, i);
  /* [16:25] */ //Histogram DCC Error and Warning Counters being nonzero
  for(int i=0; i<10; i++){
    if (dccHeader->isThisDCCErrorCounterNonZero((unsigned int)i))
	meDCCErrorAndWarnConditions_->Fill(dccid, i+1+HcalDCCHeader::SPIGOT_COUNT);
  }

  ////////// Histogram Spigot Errors from the DCCs HTR Summaries;////////////
  /* [1:8] */ //Histogram HTR Error Bits in the DCC Headers
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
      if(htrEvtN!=lastEvtN_) {meEvtNumberSynch_->Fill(slotnum,cratenum);
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
      if(htrBCN!=lastBCN_) {meBCNSynch_->Fill(slotnum,cratenum);
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




