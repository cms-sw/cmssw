#include "DQM/CastorMonitor/interface/CastorDataIntegrityMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <iostream>

//****************************************************//
//********** CastorDataIntegrityMonitor  ******************//
//********** Author: Dmytro Volyanskyy   *************//
//********** Date  : 16.03.2011 (first version) ******// 
//****************************************************//
////---- Data Integrity checks 
////---- similar to HcalDataIntegrityTask


//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorDataIntegrityMonitor::CastorDataIntegrityMonitor() {
}


//==================================================================//
//======================= Destructor ===============================//
//==================================================================//
CastorDataIntegrityMonitor::~CastorDataIntegrityMonitor() {
}


//==================================================================//
//=========================== reset  ===============================//
//==================================================================//
void CastorDataIntegrityMonitor::reset(){
 
}

//==================================================================//
//=========================== cleanup  =============================//
//==================================================================//

void CastorDataIntegrityMonitor::cleanup(){
 
} 

//==================================================================//
//=========================== setup  ==============================//
//==================================================================//

void CastorDataIntegrityMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe)
{
 
 if(fVerbosity>0) std::cout << "CastorDataIntegrityMonitor::setup (start)" << std::endl;
 CastorBaseMonitor::setup(ps,dbe);
 baseFolder_ = rootFolder_+"CastorDataIntegrityMonitor";
 ievt_=0;
 spigotStatus=-99;
 statusSpigotDCC=-99;
 CDFProbThisDCC = false; 


  ////---- initialize array 
  for (int row=0; row<15; row++) 
    for (int col=0; col<3; col++)
        problemsSpigot[row][col] = 0;
 


 ////---- fill the vector with CASTOR FEDs
 for (int i=FEDNumbering::MINCASTORFEDID; i<=FEDNumbering:: MAXCASTORFEDID; ++i)       
      fedUnpackList_.push_back(i);

  if ( m_dbe !=NULL ) {
    m_dbe->setCurrentFolder(baseFolder_);
    ////---- define monitor elements
    meEVT_      = m_dbe->bookInt("Digi Task Event Number");
    fedEntries  = m_dbe->book1D("FEDEntries_CASTOR" ,"Number of Entries per CASTOR FED",3,690,693); //-- FED 693 take out for the time being
    fedFatal    = m_dbe->book1D("CASTOR FEDFatal errors"   ,"Number of Fatal Errors per CASTOR FED",3,690,693); //-- FED 693 take out for the time being
    //fedNonFatal = m_dbe->book1D("FEDNonFatal_CASTOR","Number of non-fatal errors CASTOR FED",3,690,693); //-- take it out for the time being

    //meDCCVersion =  m_dbe->bookProfile("DCC Firmware Version","DCC Firmware Version", 3, 690, 693, 256, -0.5, 255.5);
    //meDCCVersion->setAxisTitle("FED ID", 1);
    spigotStatusMap = m_dbe->book2D("CASTOR spigot status","CASTOR spigot status",15,0,15,4,690,694); //-- get some space for the legend

  }
  
   else{ 
   if(fVerbosity>0) std::cout << "CastorDigiMonitor::setup - NO DQMStore service" << std::endl; 
  }

  if(fVerbosity>0) std::cout << "CastorDigiMonitor::setup (end)" << std::endl;

 return;

}


void CastorDataIntegrityMonitor::processEvent(const FEDRawDataCollection& RawData, const HcalUnpackerReport& unpackReport, const CastorElectronicsMap& emap){

  if(fVerbosity>0) std::cout << "CastorDataIntegrityMonitor::processEvent" << std::endl;

  meEVT_->Fill(ievt_); 

  ////---- increment here
  ievt_++;

  ////---- loop over all FEDs unpacking them 
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin();i!=fedUnpackList_.end(); i++)  {
      const FEDRawData& fed = RawData.FEDData(*i);
      if (fed.size()<12) continue;
      unpack(fed,emap);
    }

  ///--- check the spigot arrays
  for (int spigot=0; spigot<15; spigot++){
    for (int dcc=0; dcc<3; dcc++){

      if( problemsSpigot[spigot][dcc] == 0)  statusSpigotDCC=1.0;
      else if( double(problemsSpigot[spigot][dcc])/double(ievt_)  < 0.05) statusSpigotDCC=0.;       
      else statusSpigotDCC=-1.0;
      ////--- fill spigotStatusMap
      spigotStatusMap->getTH2F()->SetBinContent(spigot+1,dcc+1,statusSpigotDCC);
      if(fVerbosity>0) 
      std::cout<< "==> SpigotNr:"<< spigot+1 <<" DCC_ID:"<< dcc+690 << " # problems=" << problemsSpigot[spigot][dcc]
	       << "==> ievt_:"<< ievt_ << " ratio=" << double(problemsSpigot[spigot][dcc])/double(ievt_) << " STATUS=" << statusSpigotDCC << std::endl;
    }
  }



   
   

  return;
} 


//=======================================================//
//=============== unpack CASTOR FED =====================//
//=======================================================//

void CastorDataIntegrityMonitor::unpack(const FEDRawData& raw, const CastorElectronicsMap& emap) {


  ////---- get the DCC header
  const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
  if(!dccHeader) return;

  ////---- get the DCC trailer 
  unsigned char* trailer_ptr = (unsigned char*) (raw.data()+raw.size()-sizeof(uint64_t));
  FEDTrailer trailer = FEDTrailer(trailer_ptr);

  dccid=dccHeader->getSourceId();

  HcalHTRData htr;

  ////----  problems with the Common Data Format (CDF) compliance
  CDFProbThisDCC = false; 


  //================ BASIC CHECKS OF DATA INTEGRITY ====================//

 
  ////---- for the time being do what HCAL does 

  /* 1 */ //There should always be a second CDF header word indicated.
  if (!dccHeader->thereIsASecondCDFHeaderWord()) 
    {
      CDFProbThisDCC = true;  
    }

  /* 2 */ //Make sure a reference CDF Version value has been recorded for this dccid
  CDFvers_it = CDFversionNumber_list.find(dccid);
  if (CDFvers_it  == CDFversionNumber_list.end()) 
    {
      CDFversionNumber_list.insert(std::pair<int,short>
				   (dccid,dccHeader->getCDFversionNumber() ) );
      CDFvers_it = CDFversionNumber_list.find(dccid);
    } // then check against it.

  if (dccHeader->getCDFversionNumber()!= CDFvers_it->second) 
    {
      CDFProbThisDCC = true;  
    }
  
  /* 3 */ //Make sure a reference CDF EventType value has been recorded for this dccid
  CDFEvT_it = CDFEventType_list.find(dccid);
  if (CDFEvT_it  == CDFEventType_list.end()) 
    {
      CDFEventType_list.insert(std::pair<int,short>
			       (dccid,dccHeader->getCDFEventType() ) );
      CDFEvT_it = CDFEventType_list.find(dccid);
    } // then check against it.
  
  if (dccHeader->getCDFEventType()!= CDFEvT_it->second) 
    {
      // On probation until safe against Orbit Gap Calibration Triggers...
       CDFProbThisDCC = true;  
    }

  /* 4 */ //There should always be a '5' in CDF Header word 0, bits [63:60]
  if (dccHeader->BOEshouldBe5Always()!=5) 
    {
      CDFProbThisDCC = true;  
    }

  /* 5 */ //There should never be a third CDF Header word indicated.
  if (dccHeader->thereIsAThirdCDFHeaderWord()) 
    {
      CDFProbThisDCC = true;  
    }

  /* 6 */ //Make sure a reference value of Reserved Bits has been recorded for this dccid

  CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  if (CDFReservedBits_it  == CDFReservedBits_list.end()) {
    CDFReservedBits_list.insert(std::pair<int,short>
				(dccid,dccHeader->getSlink64ReservedBits() ) );
    CDFReservedBits_it = CDFReservedBits_list.find(dccid);
  } // then check against it.
  
  if ((int) dccHeader->getSlink64ReservedBits()!= CDFReservedBits_it->second) 
    {
    // On probation until safe against Orbit Gap Calibration Triggers...
    //       CDFProbThisDCC = true;
    }

  /* 7 */ //There should always be 0x0 in CDF Header word 1, bits [63:60]
  if (dccHeader->BOEshouldBeZeroAlways() !=0) 
    {
      CDFProbThisDCC = true;
    }
  
  /* 8 */ //There should only be one trailer
  if (trailer.moreTrailers()) 
    {
      CDFProbThisDCC = true; 
    }
  //  if trailer.

  /* 9 */ //CDF Trailer [55:30] should be the # 64-bit words in the EvFragment
  if ((uint64_t) raw.size() != ( (uint64_t) trailer.lenght()*sizeof(uint64_t)) )  //The function name is a typo! Awesome.
    {
      CDFProbThisDCC = true; 
    }
  /*10 */ //There is a rudimentary sanity check built into the FEDTrailer class
  if (!trailer.check()) 
    {
      CDFProbThisDCC = true; 
    }

  ////---- fill fatal errors 
  if (CDFProbThisDCC) fedFatal->Fill(dccid);

  ////---- fill entries per event
  fedEntries->Fill(dccid);
	


  //================== do similar what HCALRawDataMonitor does   
  ////---- get DCC Firmware Version
  //uint64_t* dccfw= (uint64_t*) (raw.data()+(sizeof(uint64_t)*2)); //64-bit DAQ word number 2 (from 0)
  //int dcc_fw =  ( ((*dccfw)>>(6*8))&0x00000000000000FF );         //Shift right 6 bytes, get that low byte.
  //meDCCVersion->Fill(dccid,dcc_fw);
  //char TTS_state = (char)trailer.ttsBits();

  //errors per-Spigot bits from the DCC Header
  int WholeErrorList=0; 

  ////--- loop over spigots
  for(int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {

  if (!( dccHeader->getSpigotEnabled((unsigned int) spigot)) ) continue; //-- skip when not enabled

  //-- set OK here 
  spigotStatus=1;
  //-- get DCC id
  dccid=dccHeader->getSourceId();

  if (dccid==693) continue; //-- skip this one

  ////---- walk through the error list
   WholeErrorList=dccHeader->getLRBErrorBits((unsigned int) spigot);
    if (WholeErrorList!=0) {

      if ((WholeErrorList>>0)&0x01) {
	if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: error==> " << std::endl;
        spigotStatus=-1; 
        ++problemsSpigot[spigot][dccid-690];
      }

      if (((WholeErrorList>>1)&0x01)!=0)  {
        if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: error ==> Uncorrected Error" << std::endl; 
           if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
           spigotStatus=-1;
       }

      if (((WholeErrorList>>2)&0x01)!=0)  {
	if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: error ==> Truncated data coming into LRB" << std::endl; 
          if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
          spigotStatus=-1;
      }

      if (((WholeErrorList>>3)&0x01)!=0)  {
	if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor: error ==>: FIFO Overflow" << std::endl; 
        if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
        spigotStatus=-1;
      }

      if (((WholeErrorList>>4)&0x01)!=0)  {
	if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: error ==> (EvN Mismatch), htr payload metadeta" << std::endl; 
         if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
         spigotStatus=-1;
      }

      if (((WholeErrorList>>5)&0x01)!=0)  {
	if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: error ==> STatus: hdr/data/trlr error" << std::endl; 
        if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
        spigotStatus=-1;
      }

      if (((WholeErrorList>>6)&0x01)!=0)  {
	if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: error ==> ODD 16-bit word count from HT error" << std::endl;  
        if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
        spigotStatus=-1;
      }

    }

    ////---- check HTR problems

    if (!dccHeader->getSpigotPresent((unsigned int) spigot)){
          if(fVerbosity>0)  std::cout <<"CastorDataIntegrityMonitor:: HTR Problem: Spigot Not Present"<<std::endl;
          if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
          spigotStatus=-1;
    }
   else {
      if ( dccHeader->getSpigotDataTruncated((unsigned int) spigot)) {
     	if(fVerbosity>0)  std::cout <<"CastorDataIntegrityMonitor:: HTR Problem: Spigot Data Truncated"<<std::endl;
        if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
        spigotStatus=-1;
      }
      if ( dccHeader->getSpigotCRCError((unsigned int) spigot)) {
	if(fVerbosity>0)  std::cout <<"CastorDataIntegrityMonitor:: HTR Problem: Spigot CRC Error"<<std::endl; 
        if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
        spigotStatus=-1;
      }
     if (dccHeader->getSpigotDataLength(spigot) <(unsigned long)4) {
      if(fVerbosity>0)  std::cout <<"CastorDataIntegrityMonitor:: HTR Problem: Spigot Data Length too small"<<std::endl; 
      if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];  
      spigotStatus=-1;
     }  

     if (dccHeader->getSpigotData(spigot,htr,raw.size())==-1) {
	if(fVerbosity>0)  std::cout<< "CastorDataIntegrityMonitor:: Invalid HTR data (data beyond payload size) observed on spigot " << spigot 
		 << " of DCC with source id " << dccHeader->getSourceId()<< std::endl;
      if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];
       spigotStatus=-1;
     }
	
    if (!htr.check()) {
      if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: Invalid HTR data observed on spigot " << spigot << " of DCC with source id " << dccHeader->getSourceId() << std::endl;
     if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];
     spigotStatus=-1;
    }

    if (htr.isHistogramEvent()) {
     if(fVerbosity>0)  std::cout << "CastorDataIntegrityMonitor:: Histogram data passed to non-histogram unpacker on spigot " << spigot << " of DCC with source id " 
     << dccHeader->getSourceId() << std::endl; 
     if(spigotStatus>0) ++problemsSpigot[spigot][dccid-690];
     spigotStatus=-1;
    }
  
   }
    
  } //-- end of loop over spigots

  return;
} 
