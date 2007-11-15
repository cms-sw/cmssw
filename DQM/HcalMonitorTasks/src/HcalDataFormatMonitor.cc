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

  firstFED_ = FEDNumbering::getHcalFEDIds().first;
  for (int i=FEDNumbering::getHcalFEDIds().first; i<=FEDNumbering::getHcalFEDIds().second; i++)
    fedUnpackList_.push_back(i);

  prtlvl_ = ps.getUntrackedParameter<int>("dfPrtLvl");

  if ( m_dbe ) {
    m_dbe->setCurrentFolder(baseFolder_);
    
    meEVT_ = m_dbe->bookInt("Data Format Task Event Number");
    meEVT_->Fill(ievt_);
    
    char* type = "Spigot Format Errors";
    meSpigotFormatErrors_=  m_dbe->book1D(type,type,500,-1,999);
    type = "Bad Quality Digis";
    meBadQualityDigis_=  m_dbe->book1D(type,type,500,-1,999);
    type = "Unmapped Digis";
    meUnmappedDigis_=  m_dbe->book1D(type,type,500,-1,999);
    type = "Unmapped Trigger Primitive Digis";
    meUnmappedTPDigis_=  m_dbe->book1D(type,type,500,-1,999);
    type = "FED Error Map";
    meFEDerrorMap_ = m_dbe->book1D(type,type,33,699.5,732.5);
    type = "Evt Number Out-of-Synch";
     meEvtNumberSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    type = "BCN Not Constant";
    meBCNSynch_= m_dbe->book2D(type,type,40,-0.25,19.75,18,-0.5,17.5);
    type = "BCN";
    meBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);

    type = "BCN Check";
    meBCNCheck_ = m_dbe->book1D(type,type,501,-250.5,250.5);
    type = "EvtN Check";
    meEvtNCheck_ = m_dbe->book1D(type,type,601,-300.5,300.5);

    type = "BCN of Fiber Orbit Message";
    meFibBCN_ = m_dbe->book1D(type,type,3564,-0.5,3563.5);

    // Firmware version
    type = "HTR Firmware Version";
    meFWVersion_ = m_dbe->book2D(type,type ,256,-0.5,255.5,18,-0.5,17.5);

    int maxbits = 16;//Look at all bits
    type = "HTR Error Word by Crate";
    meErrWdCrate_ = m_dbe->book2D(type,type,18,-0.5,17.5,maxbits,-0.5,maxbits-0.5);

    type = "HTR Error Word - Crate 0";
    meCrate0HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 1";
    meCrate1HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 2";
    meCrate2HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 3";
    meCrate3HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 4";
    meCrate4HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 5";
    meCrate5HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 6";
    meCrate6HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 7";
    meCrate7HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 8";
    meCrate8HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 9";
    meCrate9HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 10";
    meCrate10HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 11";
    meCrate11HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 12";
    meCrate12HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 13";
    meCrate13HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 14";
    meCrate14HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 15";
    meCrate15HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 16";
    meCrate16HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    type = "HTR Error Word - Crate 17";
    meCrate17HTRErr_ = m_dbe->book2D(type,type,40,-0.25,19.75,maxbits,-0.5,maxbits-0.5);
    
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
  
  lastEvtN_ = -1;
  lastBCN_ = -1;
  for (vector<int>::const_iterator i=fedUnpackList_.begin();
       i!=fedUnpackList_.end(); i++) {
    const FEDRawData& fed = rawraw.FEDData(*i);
    if (fed.size()==0) continue;
    if (fed.size()<8*3) continue;
    unpack(fed,emap);
  }
  
  meSpigotFormatErrors_->Fill(report.spigotFormatErrors());
  meBadQualityDigis_->Fill(report.badQualityDigis());
  meUnmappedDigis_->Fill(report.unmappedDigis());
  meUnmappedTPDigis_->Fill(report.unmappedTPDigis());
  
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

void HcalDataFormatMonitor::unpack(const FEDRawData& raw, const
				   HcalElectronicsMap& emap){
  
  // get the DataFormat header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;
  int dccid=dccHeader->getSourceId();
  unsigned long dccEvtNum = dccHeader->getDCCEventNumber();
  int dccBCN = dccHeader->getBunchId();

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

    if (prtlvl_!=0) HTRPrint(htr,prtlvl_);

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
      if(htrEvtN!=refEvtNum) {meEvtNumberSynch_->Fill(slotnum,cratenum);
      if (prtlvl_ == 1)cout << "++++ Evt # out of sync, ref, this HTR: "<< refEvtNum << "  "<<htrEvtN <<endl;}
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
      if (prtlvl_==1)cout << "++++ BCN # out of sync, ref, this HTR: "<< refBCN << "  "<<htrBCN <<endl;}
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
    printf("Crate,Slot,ErrWord,Evt#,BCN:  %3i %4.1f %6X %7i %4X", cratenum,slotnum,htr.getErrorsWord(),htr.getL1ANumber(),htr.getBunchNumber());
    printf(" DLLunlk,TTCrdy:%2i %2i \n",htr.getDLLunlock(),htr.getTTCready());
  }
  else if (prtlvl == 2){
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    printf("Crate, Slot:%3i %4.1f", cratenum,slotnum);
    printf("  Ext Hdr: %4X %4X %4X %4X %4X %4X %4X %4X \n",htr.getExtHdr1(),htr.getExtHdr2(),htr.getExtHdr3(),htr.getExtHdr4(),htr.getExtHdr5(),htr.getExtHdr6(),htr.getExtHdr7(),htr.getExtHdr8());
}

  else if (prtlvl == 3){
    int cratenum = htr.readoutVMECrateId();
    float slotnum = htr.htrSlot() + 0.5*htr.htrTopBottom();
    printf("Crate, Slot:%3i %4.1f", cratenum,slotnum);
    printf(" FibOrbMsgBCNs: %4X %4X %4X %4X %4X %4X %4X %4X \n",htr.getFib1OrbMsgBCN(),htr.getFib2OrbMsgBCN(),htr.getFib3OrbMsgBCN(),htr.getFib4OrbMsgBCN(),htr.getFib5OrbMsgBCN(),htr.getFib6OrbMsgBCN(),htr.getFib7OrbMsgBCN(),htr.getFib8OrbMsgBCN());
  }

return;
}




