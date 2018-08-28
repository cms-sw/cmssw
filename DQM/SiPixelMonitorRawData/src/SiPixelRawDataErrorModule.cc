#include "DQM/SiPixelMonitorRawData/interface/SiPixelRawDataErrorModule.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <cstdlib>
#include <sstream>
 
// Data Formats
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
 
using namespace std;
 
const int SiPixelRawDataErrorModule::LINK_bits = 6;
const int SiPixelRawDataErrorModule::ROC_bits  = 5;
const int SiPixelRawDataErrorModule::DCOL_bits = 5;
const int SiPixelRawDataErrorModule::PXID_bits = 8;
const int SiPixelRawDataErrorModule::ADC_bits  = 8;
const int SiPixelRawDataErrorModule::DataBit_bits = 1;
 
const int SiPixelRawDataErrorModule::ADC_shift  = 0;
const int SiPixelRawDataErrorModule::PXID_shift = ADC_shift + ADC_bits;
const int SiPixelRawDataErrorModule::DCOL_shift = PXID_shift + PXID_bits;
const int SiPixelRawDataErrorModule::ROC_shift  = DCOL_shift + DCOL_bits;
const int SiPixelRawDataErrorModule::LINK_shift = ROC_shift + ROC_bits;
const int SiPixelRawDataErrorModule::DB0_shift = 0;
const int SiPixelRawDataErrorModule::DB1_shift = DB0_shift + DataBit_bits;
const int SiPixelRawDataErrorModule::DB2_shift = DB1_shift + DataBit_bits;
const int SiPixelRawDataErrorModule::DB3_shift = DB2_shift + DataBit_bits;
const int SiPixelRawDataErrorModule::DB4_shift = DB3_shift + DataBit_bits;
const int SiPixelRawDataErrorModule::DB5_shift = DB4_shift + DataBit_bits;
const int SiPixelRawDataErrorModule::DB6_shift = DB5_shift + DataBit_bits;
const int SiPixelRawDataErrorModule::DB7_shift = DB6_shift + DataBit_bits;
 
const uint32_t SiPixelRawDataErrorModule::LINK_mask = ~(~uint32_t(0) << LINK_bits);
const uint32_t SiPixelRawDataErrorModule::ROC_mask  = ~(~uint32_t(0) << ROC_bits);
const uint32_t SiPixelRawDataErrorModule::DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
const uint32_t SiPixelRawDataErrorModule::PXID_mask = ~(~uint32_t(0) << PXID_bits);
const uint32_t SiPixelRawDataErrorModule::ADC_mask  = ~(~uint32_t(0) << ADC_bits);
const uint32_t SiPixelRawDataErrorModule::DataBit_mask = ~(~uint32_t(0) << DataBit_bits);
 
const int SiPixelRawDataErrorModule::TRLRBGN_bits = 32;
const int SiPixelRawDataErrorModule::EVTLGT_bits  = 24;
const int SiPixelRawDataErrorModule::TRLREND_bits = 8;
 
const int SiPixelRawDataErrorModule::TRLRBGN_shift = 0;
const int SiPixelRawDataErrorModule::EVTLGT_shift  = TRLRBGN_shift + TRLRBGN_bits;
const int SiPixelRawDataErrorModule::TRLREND_shift = EVTLGT_shift + EVTLGT_bits;
 
const unsigned long long SiPixelRawDataErrorModule::TRLREND_mask = ~(~0ULL << TRLREND_bits);
const unsigned long long SiPixelRawDataErrorModule::EVTLGT_mask  = ~(~0ULL << EVTLGT_bits);
const unsigned long long SiPixelRawDataErrorModule::TRLRBGN_mask = ~(~0ULL << TRLRBGN_bits);
//
// Constructors
//
SiPixelRawDataErrorModule::SiPixelRawDataErrorModule() : 
  id_(0),
  ncols_(416),
  nrows_(160) 
{ 
  _debug_ = false;
}
//
SiPixelRawDataErrorModule::SiPixelRawDataErrorModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 
  _debug_ = false;
}
//
SiPixelRawDataErrorModule::SiPixelRawDataErrorModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 
  _debug_ = false;
} 
//
// Destructor
//
SiPixelRawDataErrorModule::~SiPixelRawDataErrorModule() {}
//
// Book histograms for errors with detId
//
void SiPixelRawDataErrorModule::book(const edm::ParameterSet& iConfig, DQMStore::IBooker & iBooker, int type, bool isUpgrade) {}
//
// Fill histograms
//
int SiPixelRawDataErrorModule::fill(const edm::DetSetVector<SiPixelRawDataError>& input, std::map<std::string,MonitorElement**> *meMapFEDs, bool modon, bool ladon, bool bladeon) {
  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
   
  // Get DQM interface
   
  unsigned int numberOfSeriousErrors = 0;
   
  edm::DetSetVector<SiPixelRawDataError>::const_iterator isearch = input.find(id_); // search  errors of detid
   
  if( isearch != input.end() ) {  // Not at empty iterator
    // Look at errors now
    edm::DetSet<SiPixelRawDataError>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      int FedId = di->getFedId();                  // FED the error came from
      int chanNmbr = 0;
      int errorType = di->getType();               // type of error
      int TBMType=-1; int TBMMessage=-1; int evtSize=-1; int evtNbr=-1; int fullType=-1;
      bool notReset = true;
      const int LINK_bits = 6;
      const int LINK_shift = 26;
      const uint32_t LINK_mask = ~(~(uint32_t)(0) << LINK_bits);
 
      if(modon){
	if(errorType == 32 || errorType == 33 || errorType == 34) {
 	  long long errorWord = di->getWord64();     // for 64-bit error words
 	  chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
 	  if(errorType == 34) evtSize = (errorWord >> EVTLGT_shift) & EVTLGT_mask;
	} else {
 	  uint32_t errorWord = di->getWord32();      // for 32-bit error words
 	  chanNmbr = (errorWord >> LINK_shift) & LINK_mask; // default way to get channel number.  Only different for case 29 below.
 	  switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
 	  case(28) : {
 	    int NFa = (errorWord >> DB0_shift) & DataBit_mask;
 	    int NFb = (errorWord >> DB1_shift) & DataBit_mask;
 	    int NFc = (errorWord >> DB2_shift) & DataBit_mask;
 	    int NFd = (errorWord >> DB3_shift) & DataBit_mask;
 	    int NFe = (errorWord >> DB4_shift) & DataBit_mask;
 	    int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int L1A = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (NFa==1) {fullType = 1; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFb==1) {fullType = 2; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFc==1) {fullType = 3; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFd==1) {fullType = 4; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFe==1) {fullType = 5; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NF2==1) {fullType = 6; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (L1A==1) {fullType = 7; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    chanNmbr = 0;  // signifies channel not known
 	    break; }
 	  case(29) : {
 	    int CH1 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int CH2 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int CH3 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int CH4 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int CH5 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int BLOCK_bits      = 3;
 	    int BLOCK_shift     = 8;
 	    uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
 	    int BLOCK = (errorWord >> BLOCK_shift) & BLOCK_mask;
 	    int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
 	    if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
 	    else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
 	    if ((chanNmbr<1)||(chanNmbr>36)) chanNmbr=0;  // signifies unexpected result
 	    break; }
 	  case(30) : {
 	    int T0 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int T1 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int T2 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int T3 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int T4 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int T5 = (errorWord >> DB5_shift) & DataBit_mask;
 	    int T6 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int T7 = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (T0==1) TBMMessage=0;
 	    if (T1==1) TBMMessage=1;
 	    if (T2==1) TBMMessage=2;
 	    if (T3==1) TBMMessage=3;
 	    if (T4==1) TBMMessage=4;
 	    if (T5==1) TBMMessage=5;
 	    if (T6==1) TBMMessage=6;
 	    if (T7==1) TBMMessage=7;
	    if(TBMMessage==5 || TBMMessage==6) notReset=false;
 	    int StateMach_bits      = 4;
 	    int StateMach_shift     = 8;
 	    uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
 	    int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
 	    switch(StateMach) {
 	    case(0) : {
 	      TBMType = 0;
 	      break; }
 	    case(1) : case(9) : {
		TBMType = 1;
		break; }
 	    case(2) : case(4) : case(6) : {
		TBMType = 2;
		break; }
 	    case(8) : {
 	      TBMType = 3;
 	      break; }
 	    default : TBMType = 4;
 	    };
 	    break; }
 	  case(31) : {
 	    evtNbr = (errorWord >> ADC_shift) & ADC_mask;
 	    break; }
 	  case(36) : {
 	    //ROCId = (errorWord >> ROC_shift) & ROC_mask;
 	    break; }
 	  case(37) : {
 	    //DCOLId = (errorWord >> DCOL_shift) & DCOL_mask;
 	    //PXId = (errorWord >> PXID_shift) & PXID_mask;
 	    break; }
 	  case(38) : {
 	    //ROCNmbr = (errorWord >> ROC_shift) & ROC_mask;
 	    break; }
 	  default : break;
 	  };
 	}//end if not double precision  
      }//end if modon
 
      if(ladon && barrel){
	if(errorType == 32 || errorType == 33 || errorType == 34){
 	  long long errorWord = di->getWord64();     // for 64-bit error words
 	  if(errorType == 34) evtSize = (errorWord >> EVTLGT_shift) & EVTLGT_mask;
 	  chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
	} else {
 	  uint32_t errorWord = di->getWord32();      // for 32-bit error words
 	  chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
 	  switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
 	  case(28) : {
 	    int NFa = (errorWord >> DB0_shift) & DataBit_mask;
 	    int NFb = (errorWord >> DB1_shift) & DataBit_mask;
 	    int NFc = (errorWord >> DB2_shift) & DataBit_mask;
 	    int NFd = (errorWord >> DB3_shift) & DataBit_mask;
 	    int NFe = (errorWord >> DB4_shift) & DataBit_mask;
 	    int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int L1A = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (NFa==1) {fullType = 1; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFb==1) {fullType = 2; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFc==1) {fullType = 3; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFd==1) {fullType = 4; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFe==1) {fullType = 5; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NF2==1) {fullType = 6; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (L1A==1) {fullType = 7; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    chanNmbr = 0;
 	    break; }
 	  case(29) : {
 	    int CH1 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int CH2 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int CH3 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int CH4 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int CH5 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int BLOCK_bits      = 3;
 	    int BLOCK_shift     = 8;
 	    uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
 	    int BLOCK = (errorWord >> BLOCK_shift) & BLOCK_mask;
 	    int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
 	    if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
 	    else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
 	    if ((chanNmbr<1)||(chanNmbr>36)) chanNmbr=0;  // signifies unexpected result
 	    break; }
 	  case(30) : {
 	    int T0 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int T1 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int T2 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int T3 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int T4 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int T5 = (errorWord >> DB5_shift) & DataBit_mask;
 	    int T6 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int T7 = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (T0==1) TBMMessage=0;
 	    if (T1==1) TBMMessage=1;
 	    if (T2==1) TBMMessage=2;
 	    if (T3==1) TBMMessage=3;
 	    if (T4==1) TBMMessage=4;
 	    if (T5==1) TBMMessage=5;
 	    if (T6==1) TBMMessage=6;
 	    if (T7==1) TBMMessage=7;
 	    int StateMach_bits      = 4;
 	    int StateMach_shift     = 8;
 	    uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
 	    int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
 	    switch(StateMach) {
 	    case(0) : {
 	      TBMType = 0;
 	      break; }
 	    case(1) : case(9) : {
		TBMType = 1;
		break; }
 	    case(2) : case(4) : case(6) : {
		TBMType = 2;
		break; }
 	    case(8) : {
 	      TBMType = 3;
 	      break; }
 	    default : TBMType = 4;
 	    };
 	    break; }
 	  case(31) : {
 	    evtNbr = (errorWord >> ADC_shift) & ADC_mask;
 	    break; }
 	  case(36) : {
 	    //int ROCId = (errorWord >> ROC_shift) & ROC_mask;
 	    break; }
 	  case(37) : {
 	    //int DCOLId = (errorWord >> DCOL_shift) & DCOL_mask;
 	    //int PXId = (errorWord >> PXID_shift) & PXID_mask;
 	    break; }
 	  case(38) : {
 	    //int ROCNmbr = (errorWord >> ROC_shift) & ROC_mask;
 	    break; }
 	  default : break;
 	  };
 	}
      }//end if ladderon
 
      if(bladeon && endcap){
	if(errorType == 32 || errorType == 33 || errorType == 34){
 	  long long errorWord = di->getWord64();     // for 64-bit error words
 	  if(errorType == 34) evtSize = (errorWord >> EVTLGT_shift) & EVTLGT_mask;
 	  chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
	} else {
 	  uint32_t errorWord = di->getWord32();      // for 32-bit error words
 	  chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
 	  switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
 	  case(28) : {
 	    int NFa = (errorWord >> DB0_shift) & DataBit_mask;
 	    int NFb = (errorWord >> DB1_shift) & DataBit_mask;
 	    int NFc = (errorWord >> DB2_shift) & DataBit_mask;
 	    int NFd = (errorWord >> DB3_shift) & DataBit_mask;
 	    int NFe = (errorWord >> DB4_shift) & DataBit_mask;
 	    int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int L1A = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (NFa==1) {fullType = 1; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFb==1) {fullType = 2; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFc==1) {fullType = 3; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFd==1) {fullType = 4; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NFe==1) {fullType = 5; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (NF2==1) {fullType = 6; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    if (L1A==1) {fullType = 7; ((*meMapFEDs)["meFullType_"][FedId])->Fill((int)fullType);}
 	    chanNmbr = 0;
 	    break; }
 	  case(29) : {
 	    int CH1 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int CH2 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int CH3 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int CH4 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int CH5 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int BLOCK_bits      = 3;
 	    int BLOCK_shift     = 8;
 	    uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
 	    int BLOCK = (errorWord >> BLOCK_shift) & BLOCK_mask;
 	    int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
 	    if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
 	    else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
 	    if ((chanNmbr<1)||(chanNmbr>36)) chanNmbr=0;  // signifies unexpected result
 	    break; }
 	  case(30) : {
 	    int T0 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int T1 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int T2 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int T3 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int T4 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int T5 = (errorWord >> DB5_shift) & DataBit_mask;
 	    int T6 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int T7 = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (T0==1) TBMMessage=0;
 	    if (T1==1) TBMMessage=1;
 	    if (T2==1) TBMMessage=2;
 	    if (T3==1) TBMMessage=3;
 	    if (T4==1) TBMMessage=4;
 	    if (T5==1) TBMMessage=5;
 	    if (T6==1) TBMMessage=6;
 	    if (T7==1) TBMMessage=7;
 	    int StateMach_bits      = 4;
 	    int StateMach_shift     = 8;
 	    uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
 	    int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
 	    switch(StateMach) {
 	    case(0) : {
 	      TBMType = 0;
 	      break; }
 	    case(1) : case(9) : {
		TBMType = 1;
		break; }
 	    case(2) : case(4) : case(6) : {
		TBMType = 2;
		break; }
 	    case(8) : {
 	      TBMType = 3;
 	      break; }
 	    default : TBMType = 4;
 	    };
 	    break; }
 	  case(31) : {
 	    evtNbr = (errorWord >> ADC_shift) & ADC_mask;
 	    break; }
 	  case(36) : {
 	    //int ROCId = (errorWord >> ROC_shift) & ROC_mask;
 	    break; }
 	  case(37) : {
 	    //int DCOLId = (errorWord >> DCOL_shift) & DCOL_mask;
 	    //int PXId = (errorWord >> PXID_shift) & PXID_mask;
 	    break; }
 	  case(38) : {
 	    //int ROCNmbr = (errorWord >> ROC_shift) & ROC_mask;
 	    break; }
 	  default : break;
 	  };
 	}
      }//end if bladeon
       
      // Example to mask a specific bad channel --> if(!(FedId==38&&chanNmbr==7)){
      if(!(errorType==30) || notReset){
	std::string hid;
	static const char chNfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErr_%d";
	char chNbuf[sizeof(chNfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
	sprintf(chNbuf, chNfmt, FedId, chanNmbr);
	hid = chNbuf;
	if((*meMapFEDs)["meFedChNErr_"][FedId]) (*meMapFEDs)["meFedChNErr_"][FedId]->Fill(chanNmbr);

	static const char chLfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChLErr_%d";
	char chLbuf[sizeof(chLfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
	sprintf(chLbuf, chLfmt, FedId, chanNmbr);
	hid = chLbuf;
	if((*meMapFEDs)["meFedChLErr_"][FedId]) (*meMapFEDs)["meFedChLErr_"][FedId]->setBinContent(chanNmbr+1,errorType);

	numberOfSeriousErrors++;
	int messageType = 99;
	if(errorType<30) messageType = errorType-25;
	else if(errorType>30) messageType = errorType-19;
	else if(errorType==30 && TBMMessage==0) messageType = errorType-25;
	else if(errorType==30 && TBMMessage==1) messageType = errorType-24;
	else if(errorType==30 && (TBMMessage==2 || TBMMessage==3 || TBMMessage==4)) messageType = errorType-23;
	else if(errorType==30 && TBMMessage==7) messageType = errorType-22;
	else if(errorType==30 && TBMType==1) messageType = errorType-21;
	else if(errorType==30 && TBMType==2) messageType = errorType-20;
	else if(errorType==30 && TBMType==3) messageType = errorType-19;
	if(messageType<=20){
	  static const char fmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedETypeNErr_%d";
	  char buf[sizeof(fmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
	  sprintf(buf, fmt, FedId, messageType);
	  hid = buf;
	  if((*meMapFEDs)["meFedETypeNErr_"][FedId]) (*meMapFEDs)["meFedETypeNErr_"][FedId]->Fill(messageType);
	}
      }

      (*meMapFEDs)["meNErrors_"][FedId]->Fill((int)numberOfSeriousErrors);
      (*meMapFEDs)["meTBMMessage_"][FedId]->Fill((int)TBMMessage);
      (*meMapFEDs)["meTBMType_"][FedId]->Fill((int)TBMType);
      (*meMapFEDs)["meErrorType_"][FedId]->Fill((int)errorType);
      (*meMapFEDs)["meFullType_"][FedId]->Fill((int)fullType);
      (*meMapFEDs)["meEvtNbr_"][FedId]->setBinContent(1,(int)evtNbr);
      (*meMapFEDs)["meEvtSize_"][FedId]->setBinContent(1,(int)evtSize);
    }
  }//end if not an empty iterator
  return numberOfSeriousErrors;
}
 
int SiPixelRawDataErrorModule::fillFED(const edm::DetSetVector<SiPixelRawDataError>& input, std::map<std::string,MonitorElement**> *meMapFEDs) {
  unsigned int numberOfSeriousErrors = 0;
   
  edm::DetSetVector<SiPixelRawDataError>::const_iterator isearch = input.find(0xffffffff); // search  errors of detid
  if( isearch != input.end() ) {  // Not an empty iterator
    // Look at FED errors now	
    edm::DetSet<SiPixelRawDataError>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      int FedId = di->getFedId();                  // FED the error came from
      int chanNmbr = -1;
      int errorType = 0;               // type of error
      if(FedId==static_cast<int>(id_)) {
 	errorType = di->getType();               // type of error
 	((*meMapFEDs)["meErrorType_"][id_])->Fill((int)errorType);
	bool notReset=true;
	int TBMType=-1; int TBMMessage=-1; int evtSize=-1; int fullType=-1;
	const int LINK_bits = 6;
	const int LINK_shift = 26;
	const uint32_t LINK_mask = ~(~(uint32_t)(0) << LINK_bits);
 	if((errorType == 32)||(errorType == 33)||(errorType == 34)) {
 	  long long errorWord = di->getWord64();     // for 64-bit error words
 	  chanNmbr = 0;
 	  switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
 	  case(32) : {
 	    break; }
 	  case(33) : {
 	    break; }
 	  case(34) : {
 	    evtSize = (errorWord >> EVTLGT_shift) & EVTLGT_mask;
	    ((*meMapFEDs)["meEvtSize_"][id_])->setBinContent(1,(int)evtSize);
 	    break; }
 	  default : break;
 	  };
 	} else {
 	  uint32_t errorWord = di->getWord32();      // for 32-bit error words
 	  switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
 	  case(25) : case(39) : {
	      chanNmbr = 0;
	      break; }
 	  case(28) : {
 	    int NFa = (errorWord >> DB0_shift) & DataBit_mask;
 	    int NFb = (errorWord >> DB1_shift) & DataBit_mask;
 	    int NFc = (errorWord >> DB2_shift) & DataBit_mask;
 	    int NFd = (errorWord >> DB3_shift) & DataBit_mask;
 	    int NFe = (errorWord >> DB4_shift) & DataBit_mask;
 	    int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int L1A = (errorWord >> DB7_shift) & DataBit_mask;
 	    if (NFa==1) {fullType = 1; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    if (NFb==1) {fullType = 2; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    if (NFc==1) {fullType = 3; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    if (NFd==1) {fullType = 4; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    if (NFe==1) {fullType = 5; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    if (NF2==1) {fullType = 6; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    if (L1A==1) {fullType = 7; ((*meMapFEDs)["meFullType_"][id_])->Fill((int)fullType);}
 	    chanNmbr = 0;
 	    break; }
 	  case(29) : {
 	    int CH1 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int CH2 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int CH3 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int CH4 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int CH5 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int BLOCK_bits      = 3;
 	    int BLOCK_shift     = 8;
 	    uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
 	    int BLOCK = (errorWord >> BLOCK_shift) & BLOCK_mask;
 	    int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
 	    if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
 	    else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
 	    if ((chanNmbr<1)||(chanNmbr>36)) chanNmbr=0;  // signifies unexpected result
 	    break; }
 	  case(30) : {
 	    int T0 = (errorWord >> DB0_shift) & DataBit_mask;
 	    int T1 = (errorWord >> DB1_shift) & DataBit_mask;
 	    int T2 = (errorWord >> DB2_shift) & DataBit_mask;
 	    int T3 = (errorWord >> DB3_shift) & DataBit_mask;
 	    int T4 = (errorWord >> DB4_shift) & DataBit_mask;
 	    int T5 = (errorWord >> DB5_shift) & DataBit_mask;
 	    int T6 = (errorWord >> DB6_shift) & DataBit_mask;
 	    int T7 = (errorWord >> DB7_shift) & DataBit_mask;
 	    if(!(FedId==38&&chanNmbr==7)){
 	      if (T0==1) { TBMMessage=0; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T1==1) { TBMMessage=1; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T2==1) { TBMMessage=2; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T3==1) { TBMMessage=3; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T4==1) { TBMMessage=4; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T5==1) { TBMMessage=5; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T6==1) { TBMMessage=6; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	      if (T7==1) { TBMMessage=7; ((*meMapFEDs)["meTBMMessage_"][id_])->Fill((int)TBMMessage); }
 	    }
	    if(TBMMessage==5 || TBMMessage==6) notReset=false;
 	    int StateMach_bits      = 4;
 	    int StateMach_shift     = 8;
 	    uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
 	    int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
 	    switch(StateMach) {
 	    case(0) : {
 	      TBMType = 0;
 	      break; }
 	    case(1) : case(9) : {
		TBMType = 1;
		break; }
 	    case(2) : case(4) : case(6) : {
		TBMType = 2;
		break; }
 	    case(8) : {
 	      TBMType = 3;
 	      break; }
 	    default : TBMType = 4;
 	    };
 	    if(!(FedId==38&&chanNmbr==7)) ((*meMapFEDs)["meTBMType_"][id_])->Fill((int)TBMType);
 	    chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
 	    break; }
 	  case(31) : {
 	    int evtNbr = (errorWord >> ADC_shift) & ADC_mask;
 	    if(!(FedId==38&&chanNmbr==7))((*meMapFEDs)["meEvtNbr_"][id_])->setBinContent(1,(int)evtNbr);
 	    chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
 	    break; }
 	  case(35) : case(36) : case(37) : case(38) : {
	      chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
	      break; }
 	  default : break;
 	  };
 	}// end if errorType

	if(!(errorType==30) || notReset){
	  std::string hid;
	  static const char chNfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErr_%d";
	  char chNbuf[sizeof(chNfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
	  sprintf(chNbuf, chNfmt, FedId, chanNmbr);
	  hid = chNbuf;
	  if((*meMapFEDs)["meFedChNErr_"][id_]) (*meMapFEDs)["meFedChNErr_"][id_]->Fill(chanNmbr);

	  static const char chLfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChLErr_%d";
	  char chLbuf[sizeof(chLfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
	  sprintf(chLbuf, chLfmt, FedId, chanNmbr);
	  hid = chLbuf;
	  if((*meMapFEDs)["meFedChLErr_"][id_]) (*meMapFEDs)["meFedChLErr_"][id_]->setBinContent(chanNmbr+1,errorType);

	  numberOfSeriousErrors++;
	  int messageType = 99;
	  if(errorType<30) messageType = errorType-25;
	  else if(errorType>30) messageType = errorType-19;
	  else if(errorType==30 && TBMMessage==0) messageType = errorType-25;
	  else if(errorType==30 && TBMMessage==1) messageType = errorType-24;
	  else if(errorType==30 && (TBMMessage==2 || TBMMessage==3 || TBMMessage==4)) messageType = errorType-23;
	  else if(errorType==30 && TBMMessage==7) messageType = errorType-22;
	  else if(errorType==30 && TBMType==1) messageType = errorType-21;
	  else if(errorType==30 && TBMType==2) messageType = errorType-20;
	  else if(errorType==30 && TBMType==3) messageType = errorType-19;
	  if(messageType<=20){
	    static const char fmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedETypeNErr_%d";
	    char buf[sizeof(fmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
	    sprintf(buf, fmt, FedId, messageType);
	    hid = buf;
	    if((*meMapFEDs)["meFedETypeNErr_"][id_]) (*meMapFEDs)["meFedETypeNErr_"][id_]->Fill(messageType);
	  }
	}//end if not 30 || notReset
      }//end if
    }//end for
    if(numberOfSeriousErrors>0) ((*meMapFEDs)["meNErrors_"][id_])->Fill((float)numberOfSeriousErrors);
  }// end if not an empty iterator
  return numberOfSeriousErrors;
}
