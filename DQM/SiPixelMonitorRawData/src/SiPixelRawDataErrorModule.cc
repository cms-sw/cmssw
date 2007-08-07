#include "DQM/SiPixelMonitorRawData/interface/SiPixelRawDataErrorModule.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
// Framework
#include "FWCore/Framework/interface/ESHandle.h"
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

const int SiPixelRawDataErrorModule::ADC_shift  = 0;
const int SiPixelRawDataErrorModule::PXID_shift = ADC_shift + ADC_bits;
const int SiPixelRawDataErrorModule::DCOL_shift = PXID_shift + PXID_bits;
const int SiPixelRawDataErrorModule::ROC_shift  = DCOL_shift + DCOL_bits;
const int SiPixelRawDataErrorModule::LINK_shift = ROC_shift + ROC_bits;

const uint32_t SiPixelRawDataErrorModule::LINK_mask = ~(~uint32_t(0) << LINK_bits);
const uint32_t SiPixelRawDataErrorModule::ROC_mask  = ~(~uint32_t(0) << ROC_bits);
const uint32_t SiPixelRawDataErrorModule::DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
const uint32_t SiPixelRawDataErrorModule::PXID_mask = ~(~uint32_t(0) << PXID_bits);
const uint32_t SiPixelRawDataErrorModule::ADC_mask  = ~(~uint32_t(0) << ADC_bits);

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
  meErrorType_ = theDMBE->book1D(hid,"Type of errors",39,25.,39.);
  meErrorType_->setAxisTitle("Type of errors",1);
  // For error type 30, the type of problem encoded in the TBM error trailer
  // 1 = FSM errors, 2 = invalid # of ROCs, 3 = data stream too long, 4 = unexpected
  hid = theHistogramId->setHistoId("TBMType",id_);
  meTBMType_ = theDMBE->book1D(hid,"Type of TBM trailer",5,1.,5.);
  meTBMType_->setAxisTitle("TBM Type",1);
  // For error type 31, the event number of the TBM header with the error
  hid = theHistogramId->setHistoId("EvtNbr",id_);
  meEvtNbr_ = theDMBE->book1D(hid,"Event number for error type 31",256,0.,256.);
  meEvtNbr_->setAxisTitle("Event number",1);
  // For error type 36, the invalid ROC number
  hid = theHistogramId->setHistoId("ROCId",id_);
  meROCId_ = theDMBE->book1D(hid,"ROC number for error type 36",25,0.,25.);
  meROCId_->setAxisTitle("ROC Id",1);
  // For error type 37, the invalid dcol values
  hid = theHistogramId->setHistoId("DCOLId",id_);
  meDCOLId_ = theDMBE->book1D(hid,"DCOL address for error type 37",32,0.,32.);
  meDCOLId_->setAxisTitle("DCOL address",1);
  // For error type 37, the invalid dcol values
  hid = theHistogramId->setHistoId("PXId",id_);
  mePXId_ = theDMBE->book1D(hid,"Pixel address for error type 37",256,0.,256.);
  mePXId_->setAxisTitle("Pixel address",1);
  // For error type 38, the ROC that is being read out of order
  hid = theHistogramId->setHistoId("ROCNmbr",id_);
  meROCNmbr_ = theDMBE->book1D(hid,"ROC number for error type 38",25,0.,25.);
  meROCNmbr_->setAxisTitle("ROC number on DetUnit",1);
  delete theHistogramId;
}
//
// Book histograms for errors without detId
//
void SiPixelRawDataErrorModule::bookAlt(const edm::ParameterSet& iConfig) {

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  // Types of errors
  hid = theHistogramId->setHistoId("errorType",id_);
  meErrorType_ = theDMBE->book1D(hid,"Type of errors",39,25.,39.);
  meErrorType_->setAxisTitle("Type of errors",1);
  // Type of FIFO full (errorType = 28).  FIFO 1 is 1-5 (where fullType = channel of FIFO 1), 
  // fullType = 6 signifies FIFO 2 nearly full, 7 signifies trigger FIFO nearly full, 8 
  // indicates an unexpected result
  hid = theHistogramId->setHistoId("fullType",id_);
  meFullType_ = theDMBE->book1D(hid,"Type of FIFO full",9,1.,9.);
  meFullType_->setAxisTitle("FIFO type",1);
  // For errorType = 29, channel with timeout error (6 indicates unexpected result)
  hid = theHistogramId->setHistoId("chanNmbr",id_);
  meChanNmbr_ = theDMBE->book1D(hid,"Timeout Channel",65,1.,65.);
  meChanNmbr_->setAxisTitle("Channel number",1);
  // For errorType = 34, datastream size according to error word
  hid = theHistogramId->setHistoId("evtSize",id_);
  meEvtSize_ = theDMBE->book1D(hid,"Header Datastream Size",500,0.,500.);
  meEvtSize_->setAxisTitle("Number of words",1);
  // For errorType = 35, the bad channel number
  hid = theHistogramId->setHistoId("linkId",id_);
  meLinkId_ = theDMBE->book1D(hid,"Invalid Channel Number",65,1.,65.);
  meLinkId_->setAxisTitle("Channel number",1);

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
	case(28) : {
	  int FIFObits = (errorWord >> ADC_shift) & ADC_mask;
	  int fullType;
	  switch(FIFObits) {
	  case(1) : {
	    fullType = 1;
	    break; }
	  case(2) : {
	    fullType = 2;
	    break; }
	  case(4) : {
	    fullType = 3;
	    break; }
	  case(8) : {
	    fullType = 4;
	    break; }
	  case(16) : {
	    fullType = 5;
	    break; }
	  case(64) : {
	    fullType = 6;
	    break; }
	  case(128) : {
	    fullType = 7;
	    break; }
	  default : fullType = 8;
	  };
	  (meFullType_)->Fill((int)fullType);
	  break; }
	case(29) : {
	  int chanNmbr = (errorWord >> ADC_shift) & ADC_mask;
	  (meChanNmbr_)->Fill((int)chanNmbr);
	  break; }
	case(30) : {
	  int TBMBits = (errorWord >> PXID_shift) & PXID_mask;
	  int TBMType;
	  switch(TBMBits) {
	  case(6) : {
	    TBMType = 1;
	    break; }
	  case(8) : {
	    TBMType = 2;
	    break; }
	  case(15) : {
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
    //std::cout<<"number of errors="<<numberOfErrors<<std::endl;meROCId_
    
  }
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
