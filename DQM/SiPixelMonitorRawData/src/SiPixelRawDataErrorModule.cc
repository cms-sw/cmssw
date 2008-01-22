#include "DQM/SiPixelMonitorRawData/interface/SiPixelRawDataErrorModule.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdlib.h>

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
SiPixelRawDataErrorModule::SiPixelRawDataErrorModule() : id_(0),
					 ncols_(416),
					 nrows_(160) { }
//
SiPixelRawDataErrorModule::SiPixelRawDataErrorModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 
}
//
SiPixelRawDataErrorModule::SiPixelRawDataErrorModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 
}
//
// Destructor
//
SiPixelRawDataErrorModule::~SiPixelRawDataErrorModule() {}
//
// Book histograms for errors with detId
//
void SiPixelRawDataErrorModule::book(const edm::ParameterSet& iConfig) {

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  // Types of errors
  hid = theHistogramId->setHistoId("errorType",id_);
  meErrorType_ = theDMBE->book1D(hid,"Type of errors",14,24.5,38.5);
  meErrorType_->setAxisTitle("Type of errors",1);
  // Number of errors
  hid = theHistogramId->setHistoId("NErrors",id_);
  meNErrors_ = theDMBE->book1D(hid,"Number of errors",500,-0.5,499.5);
  meNErrors_->setAxisTitle("Number of errors",1);
  // For error type 30, the type of problem encoded in the TBM trailer
  // 0 = stack full, 1 = Pre-cal issued, 2 = clear trigger counter, 3 = sync trigger, 
  // 4 = sync trigger error, 5 = reset ROC, 6 = reset TBM, 7 = no token bit pass
  hid = theHistogramId->setHistoId("TBMMessage",id_);
  meTBMMessage_ = theDMBE->book1D(hid,"TBM trailer message",8,-0.5,7.5);
  meTBMMessage_->setAxisTitle("TBM message",1);
  // For error type 30, the type of problem encoded in the FSM bits, 0 = none
  // 1 = FSM errors, 2 = invalid # of ROCs, 3 = data stream too long, 4 = multiple
  hid = theHistogramId->setHistoId("TBMType",id_);
  meTBMType_ = theDMBE->book1D(hid,"State Machine message",5,-0.5,4.5);
  meTBMType_->setAxisTitle("FSM Type",1);
  // For error type 31, the event number of the TBM header with the error
  hid = theHistogramId->setHistoId("EvtNbr",id_);
  meEvtNbr_ = theDMBE->book1D(hid,"Event number for error type 31",256,-0.5,255.5);
  meEvtNbr_->setAxisTitle("Event number",1);
  // For error type 36, the invalid ROC number
  hid = theHistogramId->setHistoId("ROCId",id_);
  meROCId_ = theDMBE->book1D(hid,"ROC number for error type 36",25,-0.5,24.5);
  meROCId_->setAxisTitle("ROC Id",1);
  // For error type 37, the invalid dcol values
  hid = theHistogramId->setHistoId("DCOLId",id_);
  meDCOLId_ = theDMBE->book1D(hid,"DCOL address for error type 37",32,-0.5,31.5);
  meDCOLId_->setAxisTitle("DCOL address",1);
  // For error type 37, the invalid ROC values
  hid = theHistogramId->setHistoId("PXId",id_);
  mePXId_ = theDMBE->book1D(hid,"Pixel address for error type 37",256,-0.5,255.5);
  mePXId_->setAxisTitle("Pixel address",1);
  // For error type 38, the ROC that is being read out of order
  hid = theHistogramId->setHistoId("ROCNmbr",id_);
  meROCNmbr_ = theDMBE->book1D(hid,"ROC number for error type 38",25,-0.5,24.5);
  meROCNmbr_->setAxisTitle("ROC number on DetUnit",1);

  delete theHistogramId;
}
//
// Book histograms for errors with detId
//
void SiPixelRawDataErrorModule::bookFED(const edm::ParameterSet& iConfig) {

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  // Types of errors
  hid = theHistogramId->setHistoId("errorType",id_);
  meErrorType_ = theDMBE->book1D(hid,"Type of errors",14,24.5,38.5);
  meErrorType_->setAxisTitle("Type of errors",1);
  // Number of errors
  hid = theHistogramId->setHistoId("NErrors",id_);
  meNErrors_ = theDMBE->book1D(hid,"Number of errors",500,-0.5,499.5);
  meNErrors_->setAxisTitle("Number of errors",1);
  // Type of FIFO full (errorType = 28).  FIFO 1 is 1-5 (where fullType = channel of FIFO 1), 
  // fullType = 6 signifies FIFO 2 nearly full, 7 signifies trigger FIFO nearly full, 8 
  // indicates an unexpected result
  hid = theHistogramId->setHistoId("fullType",id_);
  meFullType_ = theDMBE->book1D(hid,"Type of FIFO full",7,0.5,7.5);
  meFullType_->setAxisTitle("FIFO type",1);
  // For errorType = 29, channel with timeout error (0 indicates unexpected result)
  hid = theHistogramId->setHistoId("chanNmbr",id_);
  meChanNmbr_ = theDMBE->book1D(hid,"Timeout Channel",37,-0.5,36.5);
  meChanNmbr_->setAxisTitle("Channel number",1);
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
  meEvtNbr_ = theDMBE->book1D(hid,"Event number for error type 31",256,-0.5,255.5);
  meEvtNbr_->setAxisTitle("Event number",1);
  // For errorType = 34, datastream size according to error word
  hid = theHistogramId->setHistoId("evtSize",id_);
  meEvtSize_ = theDMBE->book1D(hid,"Trailer Datastream Size",500,-0.5,499.5);
  meEvtSize_->setAxisTitle("Number of words",1);
  // For errorType = 35, the bad channel number
  hid = theHistogramId->setHistoId("linkId",id_);
  meLinkId_ = theDMBE->book1D(hid,"Invalid Channel Number",35,0.5,35.5);
  meLinkId_->setAxisTitle("FED Channel number",1);
  // For error type 36, the invalid ROC number matched to FED Channel
  hid = theHistogramId->setHistoId("Type36Hitmap",id_);
  meInvROC_ = theDMBE->book2D(hid,"Invalid ROC by Channel Number",35,1.,36.,25,0.,25.);
  meInvROC_->setAxisTitle("FED Channel Number",1);
  meInvROC_->setAxisTitle("ROC Number",2);
  
  delete theHistogramId;
}
//
// Fill histograms
//
void SiPixelRawDataErrorModule::fill(const edm::DetSetVector<SiPixelRawDataError>& input) {
  
  edm::DetSetVector<SiPixelRawDataError>::const_iterator isearch = input.find(id_); // search  errors of detid
  
  if( isearch != input.end() ) {  // Not at empty iterator
    
    unsigned int numberOfErrors = 0;
    
    // Look at errors now
    edm::DetSet<SiPixelRawDataError>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      numberOfErrors++;
      int errorType = di->getType();               // type of error
      (meErrorType_)->Fill((int)errorType);
      if((errorType == 32)||(errorType == 33)||(errorType == 34)) {
	long long errorWord = di->getWord64();     // for 64-bit error words
	if(errorType == 34) {
	  int evtSize = (errorWord >> EVTLGT_shift) & EVTLGT_mask;
	  (meEvtSize_)->Fill((float)evtSize); }
      } else {
	uint32_t errorWord = di->getWord32();      // for 32-bit error words
	switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
	case(30) : {
	  int T0 = (errorWord >> DB0_shift) & DataBit_mask;
	  int T1 = (errorWord >> DB1_shift) & DataBit_mask;
	  int T2 = (errorWord >> DB2_shift) & DataBit_mask;
	  int T3 = (errorWord >> DB3_shift) & DataBit_mask;
	  int T4 = (errorWord >> DB4_shift) & DataBit_mask;
	  int T5 = (errorWord >> DB5_shift) & DataBit_mask;
	  int T6 = (errorWord >> DB6_shift) & DataBit_mask;
	  int T7 = (errorWord >> DB7_shift) & DataBit_mask;
	  int TBMMessage;
	  if (T0==1) TBMMessage=0; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T1==1) TBMMessage=1; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T2==1) TBMMessage=2; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T3==1) TBMMessage=3; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T4==1) TBMMessage=4; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T5==1) TBMMessage=5; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T6==1) TBMMessage=6; (meTBMMessage_)->Fill((int)TBMMessage);
	  if (T7==1) TBMMessage=7; (meTBMMessage_)->Fill((int)TBMMessage);
	  int StateMach_bits      = 4;
	  int StateMach_shift     = 8;
	  uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
	  int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
	  int TBMType;
	  switch(StateMach) {
	  case(0) : {
	    TBMType = 0;
	    break; }
	  case(1) : {
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
	  (meTBMType_)->Fill((int)TBMType);
	  break; }
	case(31) : {
	  int evtNbr = (errorWord >> ADC_shift) & ADC_mask;
	  (meEvtNbr_)->Fill((int)evtNbr);
	  break; }
	case(36) : {
	  int ROCId = (errorWord >> ROC_shift) & ROC_mask;
	  (meROCId_)->Fill((float)ROCId);
	  break; }
	case(37) : {
	  int DCOLId = (errorWord >> DCOL_shift) & DCOL_mask;
	  (meDCOLId_)->Fill((float)DCOLId);
	  int PXId = (errorWord >> PXID_shift) & PXID_mask;
	  (mePXId_)->Fill((float)PXId);
	  break; }
	case(38) : {
	  int ROCNmbr = (errorWord >> ROC_shift) & ROC_mask;
	  (meROCNmbr_)->Fill((float)ROCNmbr);
	  break; }
	default : break;
	};
      }
    }
    
    (meNErrors_)->Fill((float)numberOfErrors);
    //std::cout<<"number of errors="<<numberOfErrors<<std::endl;
    
  }
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}

void SiPixelRawDataErrorModule::fillFED(const edm::DetSetVector<SiPixelRawDataError>& input) {
  
  edm::DetSetVector<SiPixelRawDataError>::const_iterator isearch = input.find(0xffffffff); // search  errors of detid
  
  if( isearch != input.end() ) {  // Not at empty iterator
    
    unsigned int numberOfErrors = 0;
    
    // Look at FED errors now
    edm::DetSet<SiPixelRawDataError>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      int FedId = di->getFedId();                  // FED the error came from

      if(FedId==static_cast<int>(id_)) {
	numberOfErrors++;
	int errorType = di->getType();               // type of error
	(meErrorType_)->Fill((int)errorType);
	if((errorType == 32)||(errorType == 33)||(errorType == 34)) {
	  long long errorWord = di->getWord64();     // for 64-bit error words
	  if(errorType == 34) {
	    int evtSize = (errorWord >> EVTLGT_shift) & EVTLGT_mask;
	    (meEvtSize_)->Fill((float)evtSize); }
	} else {
	  uint32_t errorWord = di->getWord32();      // for 32-bit error words
	  switch(errorType) {  // fill in the appropriate monitorables based on the information stored in the error word
	  case(28) : {
	    int NFa = (errorWord >> DB0_shift) & DataBit_mask;
	    int NFb = (errorWord >> DB1_shift) & DataBit_mask;
	    int NFc = (errorWord >> DB2_shift) & DataBit_mask;
	    int NFd = (errorWord >> DB3_shift) & DataBit_mask;
	    int NFe = (errorWord >> DB4_shift) & DataBit_mask;
	    int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
	    int L1A = (errorWord >> DB7_shift) & DataBit_mask;
	    int fullType;
	    if (NFa==1) fullType = 1; (meFullType_)->Fill((int)fullType);
	    if (NFb==1) fullType = 2; (meFullType_)->Fill((int)fullType);
	    if (NFc==1) fullType = 3; (meFullType_)->Fill((int)fullType);
	    if (NFd==1) fullType = 4; (meFullType_)->Fill((int)fullType);
	    if (NFe==1) fullType = 5; (meFullType_)->Fill((int)fullType);
	    if (NF2==1) fullType = 6; (meFullType_)->Fill((int)fullType);
	    if (L1A==1) fullType = 7; (meFullType_)->Fill((int)fullType);
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
	    int chanNmbr;
	    if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
	    else chanNmbr = (BLOCK-1)/2+4+localCH;
	    if ((chanNmbr<1)||(chanNmbr>36)) chanNmbr=0;  // signifies unexpected result
	    (meChanNmbr_)->Fill((int)chanNmbr);
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
	    int TBMMessage;
	    if (T0==1) TBMMessage=0; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T1==1) TBMMessage=1; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T2==1) TBMMessage=2; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T3==1) TBMMessage=3; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T4==1) TBMMessage=4; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T5==1) TBMMessage=5; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T6==1) TBMMessage=6; (meTBMMessage_)->Fill((int)TBMMessage);
	    if (T7==1) TBMMessage=7; (meTBMMessage_)->Fill((int)TBMMessage);
	    int StateMach_bits      = 4;
	    int StateMach_shift     = 8;
	    uint32_t StateMach_mask = ~(~uint32_t(0) << StateMach_bits);
	    int StateMach = (errorWord >> StateMach_shift) & StateMach_mask;
	    int TBMType;
	  switch(StateMach) {
	  case(0) : {
	    TBMType = 0;
	    break; }
	  case(1) : {
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
	    (meTBMType_)->Fill((int)TBMType);
	    break; }
	  case(31) : {
	    int evtNbr = (errorWord >> ADC_shift) & ADC_mask;
	    (meEvtNbr_)->Fill((int)evtNbr);
	    break; }
	  case(35) : {
	    int linkId = (errorWord >> LINK_shift) & LINK_mask;
	    (meLinkId_)->Fill((float)linkId);
	    break; }
	  case(36) : {
	    int ROCId = (errorWord >> ROC_shift) & ROC_mask;
	    int ChanId = (errorWord >> LINK_shift) & LINK_mask;
	    (meInvROC_)->Fill((float)ChanId,(float)ROCId);
	    break; }
	  default : break;
	  };
	}
      }
    }
    
    (meNErrors_)->Fill((float)numberOfErrors);
    //std::cout<<"number of errors="<<numberOfErrors<<std::endl;meROCId_
    
  }
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
