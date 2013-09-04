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
#include <stdlib.h>
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

const long long SiPixelRawDataErrorModule::TRLREND_mask = ~(~(long long)(0) << TRLREND_bits);
const long long SiPixelRawDataErrorModule::EVTLGT_mask  = ~(~(long long)(0) << EVTLGT_bits);
const long long SiPixelRawDataErrorModule::TRLRBGN_mask = ~(~(long long)(0) << TRLRBGN_bits);
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
void SiPixelRawDataErrorModule::book(const edm::ParameterSet& iConfig, int type, bool isUpgrade) {
}
//
// Book histograms for errors within a FED
//
void SiPixelRawDataErrorModule::bookFED(const edm::ParameterSet& iConfig) {
//std::cout<<"Entering SiPixelRawDataErrorModule::bookFED: "<<std::endl;
  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  // Types of errors
  hid = theHistogramId->setHistoId("errorType",id_);
  meErrorType_ = theDMBE->book1D(hid,"Type of errors",15,24.5,39.5);
  meErrorType_->setAxisTitle("Type of errors",1);
  // Number of errors
  hid = theHistogramId->setHistoId("NErrors",id_);
  meNErrors_ = theDMBE->book1D(hid,"Number of errors",36,0.,36.);
  meNErrors_->setAxisTitle("Number of errors",1);
  // Type of FIFO full (errorType = 28).  FIFO 1 is 1-5 (where fullType = channel of FIFO 1), 
  // fullType = 6 signifies FIFO 2 nearly full, 7 signifies trigger FIFO nearly full, 8 
  // indicates an unexpected result
  hid = theHistogramId->setHistoId("fullType",id_);
  meFullType_ = theDMBE->book1D(hid,"Type of FIFO full",7,0.5,7.5);
  meFullType_->setAxisTitle("FIFO type",1);
  // For error type 30, the type of problem encoded in the TBM trailer
  // 0 = stack full, 1 = Pre-cal issued, 2 = clear trigger counter, 3 = sync trigger, 
  // 4 = sync trigger error, 5 = reset ROC, 6 = reset TBM, 7 = no token bit pass
  hid = theHistogramId->setHistoId("TBMMessage",id_);
  meTBMMessage_ = theDMBE->book1D(hid,"TBM trailer message",8,-0.5,7.5);
  meTBMMessage_->setAxisTitle("TBM message",1);
  // For error type 30, the type of problem encoded in the TBM error trailer 0 = none
  // 1 = data stream too long, 2 = FSM errors, 3 = invalid # of ROCs, 4 = multiple
  hid = theHistogramId->setHistoId("TBMType",id_);
  meTBMType_ = theDMBE->book1D(hid,"Type of TBM trailer",5,-0.5,4.5);
  meTBMType_->setAxisTitle("TBM Type",1);
  // For error type 31, the event number of the TBM header with the error
  hid = theHistogramId->setHistoId("EvtNbr",id_);
  meEvtNbr_ = theDMBE->bookInt(hid);
  // For errorType = 34, datastream size according to error word
  hid = theHistogramId->setHistoId("evtSize",id_);
  meEvtSize_ = theDMBE->bookInt(hid);
  
  for(int j=0; j!=37; j++){
    std::stringstream temp; temp << j;
    hid = "FedChNErrArray_" + temp.str();
    meFedChNErrArray_[j] = theDMBE->bookInt(hid);
    hid = "FedChLErrArray_" + temp.str();
    meFedChLErrArray_[j] = theDMBE->bookInt(hid);
    hid = "FedETypeNErrArray_" + temp.str();
    if(j<21) meFedETypeNErrArray_[j] = theDMBE->bookInt(hid);
  }
  delete theHistogramId;
//std::cout<<"...leaving SiPixelRawDataErrorModule::bookFED. "<<std::endl;
}
//
// Fill histograms
//
int SiPixelRawDataErrorModule::fill(const edm::DetSetVector<SiPixelRawDataError>& input, bool modon, bool ladon, bool bladeon) {
//std::cout<<"Entering SiPixelRawDataErrorModule::fill: "<<std::endl;
  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  
  // Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  
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
      //bool notReset = true;
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
	    if (NFa==1) fullType = 1; (meFullType_)->Fill((int)fullType);
	    if (NFb==1) fullType = 2; (meFullType_)->Fill((int)fullType);
	    if (NFc==1) fullType = 3; (meFullType_)->Fill((int)fullType);
	    if (NFd==1) fullType = 4; (meFullType_)->Fill((int)fullType);
	    if (NFe==1) fullType = 5; (meFullType_)->Fill((int)fullType);
	    if (NF2==1) fullType = 6; (meFullType_)->Fill((int)fullType);
	    if (L1A==1) fullType = 7; (meFullType_)->Fill((int)fullType);
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
	    //if(TBMMessage==5 || TBMMessage==6) notReset=false;
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
	    if (NFa==1) fullType = 1; (meFullType_)->Fill((int)fullType);
	    if (NFb==1) fullType = 2; (meFullType_)->Fill((int)fullType);
	    if (NFc==1) fullType = 3; (meFullType_)->Fill((int)fullType);
	    if (NFd==1) fullType = 4; (meFullType_)->Fill((int)fullType);
	    if (NFe==1) fullType = 5; (meFullType_)->Fill((int)fullType);
	    if (NF2==1) fullType = 6; (meFullType_)->Fill((int)fullType);
	    if (L1A==1) fullType = 7; (meFullType_)->Fill((int)fullType);
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
	    if (NFa==1) fullType = 1; (meFullType_)->Fill((int)fullType);
	    if (NFb==1) fullType = 2; (meFullType_)->Fill((int)fullType);
	    if (NFc==1) fullType = 3; (meFullType_)->Fill((int)fullType);
	    if (NFd==1) fullType = 4; (meFullType_)->Fill((int)fullType);
	    if (NFe==1) fullType = 5; (meFullType_)->Fill((int)fullType);
	    if (NF2==1) fullType = 6; (meFullType_)->Fill((int)fullType);
	    if (L1A==1) fullType = 7; (meFullType_)->Fill((int)fullType);
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
      
      if(!(FedId==38&&chanNmbr==7)){ // mask slow channel that spits out lots of EventNumber errors even when in STDBY!
	if(errorType==29 || (errorType==30 && TBMType==1)){ // consider only TIMEOUT and OVERFLOW as serious errors right now
        //if(!(errorType==30) || notReset){
          //cout<<"ERROR: "<<errorType<<" "<<TBMType<<endl;
          std::string hid;
          static const char chNfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErrArray_%d";
          char chNbuf[sizeof(chNfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
          sprintf(chNbuf, chNfmt, FedId, chanNmbr);
          hid = chNbuf;
          meFedChNErrArray_[chanNmbr] = theDMBE->get(hid);
          if(meFedChNErrArray_[chanNmbr]) meFedChNErrArray_[chanNmbr]->Fill(meFedChNErrArray_[chanNmbr]->getIntValue()+1);

          static const char chLfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChLErrArray_%d";
          char chLbuf[sizeof(chLfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
          sprintf(chLbuf, chLfmt, FedId, chanNmbr);
          hid = chLbuf;
          meFedChLErrArray_[chanNmbr] = theDMBE->get(hid); 
          if(meFedChLErrArray_[chanNmbr]) meFedChLErrArray_[chanNmbr]->Fill(errorType); 

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
            static const char fmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedETypeNErrArray_%d";
            char buf[sizeof(fmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
            sprintf(buf, fmt, FedId, messageType);
            hid = buf;
            meFedETypeNErrArray_[messageType] = theDMBE->get(hid); 
            if(meFedETypeNErrArray_[messageType]) meFedETypeNErrArray_[messageType]->Fill(meFedETypeNErrArray_[messageType]->getIntValue()+1); 
          }
	}

        std::string currDir = theDMBE->pwd();
        static const char buf[] = "Pixel/AdditionalPixelErrors/FED_%d";
        char feddir[sizeof(buf)+2]; 
        sprintf(feddir,buf,FedId);
        theDMBE->cd(feddir);
        MonitorElement* me;
        static const char buf1[] = "Pixel/AdditionalPixelErrors/FED_%d/NErrors_siPixelDigis_%d";
        char hname1[sizeof(buf1)+4];
        sprintf(hname1,buf1,FedId,FedId);
        me = theDMBE->get(hname1);
        if(me) me->Fill((int)numberOfSeriousErrors);
        static const char buf2[] = "Pixel/AdditionalPixelErrors/FED_%d/TBMMessage_siPixelDigis_%d";
        char hname2[sizeof(buf2)+4];
        sprintf(hname2,buf2,FedId,FedId);
        me = theDMBE->get(hname2);
        if(me) me->Fill((int)TBMMessage);
        static const char buf3[] = "Pixel/AdditionalPixelErrors/FED_%d/TBMType_siPixelDigis_%d";
        char hname3[sizeof(buf3)+4];
        sprintf(hname3,buf3,FedId,FedId);
        me = theDMBE->get(hname3);
        if(me) me->Fill((int)TBMType);
        static const char buf4[] = "Pixel/AdditionalPixelErrors/FED_%d/errorType_siPixelDigis_%d";
        char hname4[sizeof(buf4)+4];
        sprintf(hname4,buf4,FedId,FedId);
        me = theDMBE->get(hname4);
        if(me) me->Fill((int)errorType);
        static const char buf5[] = "Pixel/AdditionalPixelErrors/FED_%d/fullType_siPixelDigis_%d";
        char hname5[sizeof(buf5)+4];
        sprintf(hname5,buf5,FedId,FedId);
        me = theDMBE->get(hname5);
        if(me) me->Fill((int)fullType);
        static const char buf6[] = "Pixel/AdditionalPixelErrors/FED_%d/EvtNbr_siPixelDigis_%d";
        char hname6[sizeof(buf6)+4];
        sprintf(hname6,buf6,FedId,FedId);
        me = theDMBE->get(hname6);
        if(me) me->Fill((int)evtNbr);
        static const char buf7[] = "Pixel/AdditionalPixelErrors/FED_%d/evtSize_siPixelDigis_%d";
        char hname7[sizeof(buf7)+4];
        sprintf(hname7,buf7,FedId,FedId);
        me = theDMBE->get(hname7);
        if(me) me->Fill((int)evtSize);
        theDMBE->cd(currDir);
      }
    }//end for loop over all errors on module
  }//end if not an empty iterator

  return numberOfSeriousErrors;
}

int SiPixelRawDataErrorModule::fillFED(const edm::DetSetVector<SiPixelRawDataError>& input) {
  //std::cout<<"Entering   SiPixelRawDataErrorModule::fillFED: "<<static_cast<int>(id_)<<std::endl;
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
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
	(meErrorType_)->Fill((int)errorType);
	//bool notReset=true;
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
	    if(!(FedId==38&&chanNmbr==7)) (meEvtSize_)->Fill((int)evtSize); 
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
	    if (NFa==1) fullType = 1; (meFullType_)->Fill((int)fullType);
	    if (NFb==1) fullType = 2; (meFullType_)->Fill((int)fullType);
	    if (NFc==1) fullType = 3; (meFullType_)->Fill((int)fullType);
	    if (NFd==1) fullType = 4; (meFullType_)->Fill((int)fullType);
	    if (NFe==1) fullType = 5; (meFullType_)->Fill((int)fullType);
	    if (NF2==1) fullType = 6; (meFullType_)->Fill((int)fullType);
	    if (L1A==1) fullType = 7; (meFullType_)->Fill((int)fullType);
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
	      if (T0==1) { TBMMessage=0; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T1==1) { TBMMessage=1; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T2==1) { TBMMessage=2; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T3==1) { TBMMessage=3; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T4==1) { TBMMessage=4; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T5==1) { TBMMessage=5; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T6==1) { TBMMessage=6; (meTBMMessage_)->Fill((int)TBMMessage); }
	      if (T7==1) { TBMMessage=7; (meTBMMessage_)->Fill((int)TBMMessage); }
	    }
	    //if(TBMMessage==5 || TBMMessage==6) notReset=false;
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
	    if(!(FedId==38&&chanNmbr==7)) (meTBMType_)->Fill((int)TBMType);
	    chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
	    break; }
	  case(31) : {
	    int evtNbr = (errorWord >> ADC_shift) & ADC_mask;
	    if(!(FedId==38&&chanNmbr==7))(meEvtNbr_)->Fill((int)evtNbr);
	    chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
	    break; }
	  case(35) : case(36) : case(37) : case(38) : {
	    chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
	    break; }
	  default : break;
	  };
	}// end if errorType

	//if(!(errorType==30) || notReset){
	if(errorType==29 || (errorType==30 && TBMType==1)){ // consider only TIMEOUT and OVERFLOW as serious errors right now
          if(!(FedId==38&&chanNmbr==7)){ // mask slow channel that spits out lots of EventNumber errors even when in STDBY!
            std::string hid;
            //cout<<"FEDERROR: "<<errorType<<" "<<TBMType<<endl;
            static const char chNfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErrArray_%d";
            char chNbuf[sizeof(chNfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
            sprintf(chNbuf, chNfmt, FedId, chanNmbr);
            hid = chNbuf;
            meFedChNErrArray_[chanNmbr] = theDMBE->get(hid);
            if(meFedChNErrArray_[chanNmbr]) meFedChNErrArray_[chanNmbr]->Fill(meFedChNErrArray_[chanNmbr]->getIntValue()+1);

            static const char chLfmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChLErrArray_%d";
            char chLbuf[sizeof(chLfmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
            sprintf(chLbuf, chLfmt, FedId, chanNmbr);
            hid = chLbuf;
            meFedChLErrArray_[chanNmbr] = theDMBE->get(hid); 
            if(meFedChLErrArray_[chanNmbr]) meFedChLErrArray_[chanNmbr]->Fill(errorType); 

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
              static const char fmt[] = "Pixel/AdditionalPixelErrors/FED_%d/FedETypeNErrArray_%d";
              char buf[sizeof(fmt) + 2*32]; // 32 digits is enough for up to 2^105 + sign.
              sprintf(buf, fmt, FedId, messageType);
              hid = buf;
              meFedETypeNErrArray_[messageType] = theDMBE->get(hid); 
              if(meFedETypeNErrArray_[messageType]) meFedETypeNErrArray_[messageType]->Fill(meFedETypeNErrArray_[messageType]->getIntValue()+1);
            }
	  }//end if bad channel
        }//end if not 30 || notReset
      }//end if
    }//end for
    if(numberOfSeriousErrors>0) (meNErrors_)->Fill((float)numberOfSeriousErrors);

  }// end if not an empty iterator
  
//std::cout<<"...leaving   SiPixelRawDataErrorModule::fillFED. "<<std::endl;
  return numberOfSeriousErrors;
}
