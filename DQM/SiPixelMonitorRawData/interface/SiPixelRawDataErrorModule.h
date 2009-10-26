#ifndef SiPixelMonitorRawData_SiPixelRawDataErrorModule_h
#define SiPixelMonitorRawData_SiPixelRawDataErrorModule_h
// -*- C++ -*-
//
// Package:    SiPixelMonitorRawData
// Class:      SiPixelRawDataErrorModule
// 
/**\class 

 Description: 
 Error monitoring elements for a Pixel sensor.

 Implementation:
 Contains three functions to monitor error information.  "book" creates histograms in detector units and "fill" fills the monitoring elements with input error information.  
     
*/
//
// Original Author:  Andrew York
//         Created:  
//
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/cstdint.hpp>

class SiPixelRawDataErrorModule {        

 public:

  /// Default constructor
  SiPixelRawDataErrorModule();
  /// Constructor with raw DetId
  SiPixelRawDataErrorModule(const uint32_t& id);
  /// Constructor with raw DetId and sensor size
  SiPixelRawDataErrorModule(const uint32_t& id, const int& ncols, const int& nrows);
  /// Destructor
  ~SiPixelRawDataErrorModule();

  typedef edm::DetSet<SiPixelRawDataError>::const_iterator    ErrorIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig, bool reducedSet=false, int type=0);
  /// Book FED histograms
  void bookFED(const edm::ParameterSet& iConfig);
  /// Fill histograms
  void fill(const edm::DetSetVector<SiPixelRawDataError> & input, bool reducedSet=false, bool modon=true, bool ladon=false, bool layon=false, bool phion=false, bool bladeon=false, bool diskon=false, bool ringon=false);
  /// Fill FED histograms
  void fillFED(const edm::DetSetVector<SiPixelRawDataError> & input);
  
 private:

  uint32_t id_;
  int ncols_;
  int nrows_;

  MonitorElement* meErrorType_;
  MonitorElement* meNErrors_;
  MonitorElement* meFullType_;
  MonitorElement* meChanNmbr_;
  MonitorElement* meTBMMessage_;
  MonitorElement* meTBMType_;
  MonitorElement* meEvtNbr_;
  MonitorElement* meEvtSize_;
  MonitorElement* meLinkId_;
  MonitorElement* meROCId_;
  MonitorElement* meInvROC_;
  MonitorElement* meDCOLId_;
  MonitorElement* mePXId_;
  MonitorElement* meROCNmbr_;
//  MonitorElement* meNErrorsMap_;
//  MonitorElement* meLastErrorTypeMap_;
//  MonitorElement* meErrorTypeMap_;
  MonitorElement* meFedChNErrArray_[37];
  MonitorElement* meFedChLErrArray_[37];
  MonitorElement* meFedETypeNErrArray_[15];
  
  //barrel:
  MonitorElement* meErrorTypeLad_;
  MonitorElement* meNErrorsLad_;
  MonitorElement* meFullTypeLad_;
  MonitorElement* meChanNmbrLad_;
  MonitorElement* meTBMMessageLad_;
  MonitorElement* meTBMTypeLad_;
  MonitorElement* meEvtNbrLad_;
  MonitorElement* meEvtSizeLad_;
  MonitorElement* meLinkIdLad_;
  MonitorElement* meROCIdLad_;
  MonitorElement* meInvROCLad_;
  MonitorElement* meDCOLIdLad_;
  MonitorElement* mePXIdLad_;
  MonitorElement* meROCNmbrLad_;
  
  MonitorElement* meErrorTypeLay_;
  MonitorElement* meNErrorsLay_;
  MonitorElement* meFullTypeLay_;
  MonitorElement* meChanNmbrLay_;
  MonitorElement* meTBMMessageLay_;
  MonitorElement* meTBMTypeLay_;
  MonitorElement* meEvtNbrLay_;
  MonitorElement* meEvtSizeLay_;
  MonitorElement* meLinkIdLay_;
  MonitorElement* meROCIdLay_;
  MonitorElement* meInvROCLay_;
  MonitorElement* meDCOLIdLay_;
  MonitorElement* mePXIdLay_;
  MonitorElement* meROCNmbrLay_;
  
  MonitorElement* meErrorTypePhi_;
  MonitorElement* meNErrorsPhi_;
  MonitorElement* meFullTypePhi_;
  MonitorElement* meChanNmbrPhi_;
  MonitorElement* meTBMMessagePhi_;
  MonitorElement* meTBMTypePhi_;
  MonitorElement* meEvtNbrPhi_;
  MonitorElement* meEvtSizePhi_;
  MonitorElement* meLinkIdPhi_;
  MonitorElement* meROCIdPhi_;
  MonitorElement* meInvROCPhi_;
  MonitorElement* meDCOLIdPhi_;
  MonitorElement* mePXIdPhi_;
  MonitorElement* meROCNmbrPhi_;
  
  //forward:
  MonitorElement* meErrorTypeBlade_;
  MonitorElement* meNErrorsBlade_;
  MonitorElement* meFullTypeBlade_;
  MonitorElement* meChanNmbrBlade_;
  MonitorElement* meTBMMessageBlade_;
  MonitorElement* meTBMTypeBlade_;
  MonitorElement* meEvtNbrBlade_;
  MonitorElement* meEvtSizeBlade_;
  MonitorElement* meLinkIdBlade_;
  MonitorElement* meROCIdBlade_;
  MonitorElement* meInvROCBlade_;
  MonitorElement* meDCOLIdBlade_;
  MonitorElement* mePXIdBlade_;
  MonitorElement* meROCNmbrBlade_;

  MonitorElement* meErrorTypeDisk_;
  MonitorElement* meNErrorsDisk_;
  MonitorElement* meFullTypeDisk_;
  MonitorElement* meChanNmbrDisk_;
  MonitorElement* meTBMMessageDisk_;
  MonitorElement* meTBMTypeDisk_;
  MonitorElement* meEvtNbrDisk_;
  MonitorElement* meEvtSizeDisk_;
  MonitorElement* meLinkIdDisk_;
  MonitorElement* meROCIdDisk_;
  MonitorElement* meInvROCDisk_;
  MonitorElement* meDCOLIdDisk_;
  MonitorElement* mePXIdDisk_;
  MonitorElement* meROCNmbrDisk_;

  MonitorElement* meErrorTypeRing_;
  MonitorElement* meNErrorsRing_;
  MonitorElement* meFullTypeRing_;
  MonitorElement* meChanNmbrRing_;
  MonitorElement* meTBMMessageRing_;
  MonitorElement* meTBMTypeRing_;
  MonitorElement* meEvtNbrRing_;
  MonitorElement* meEvtSizeRing_;
  MonitorElement* meLinkIdRing_;
  MonitorElement* meROCIdRing_;
  MonitorElement* meInvROCRing_;
  MonitorElement* meDCOLIdRing_;
  MonitorElement* mePXIdRing_;
  MonitorElement* meROCNmbrRing_;
  

  static const int LINK_bits,  ROC_bits,  DCOL_bits,  PXID_bits,  ADC_bits, DataBit_bits, TRLRBGN_bits, EVTLGT_bits, TRLREND_bits;
  static const int LINK_shift, ROC_shift, DCOL_shift, PXID_shift, ADC_shift, DB0_shift, DB1_shift, DB2_shift, DB3_shift, DB4_shift, DB5_shift, DB6_shift, DB7_shift, TRLRBGN_shift, EVTLGT_shift, TRLREND_shift;
  static const uint32_t LINK_mask, ROC_mask, DCOL_mask, PXID_mask, ADC_mask, DataBit_mask;
  static const long long TRLRBGN_mask, EVTLGT_mask, TRLREND_mask;
};
#endif
