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
  void book(const edm::ParameterSet& iConfig);
  /// Book FED histograms
  void bookFED(const edm::ParameterSet& iConfig);
  /// Fill histograms
  void fill(const edm::DetSetVector<SiPixelRawDataError> & input);
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
  MonitorElement* meDCOLId_;
  MonitorElement* mePXId_;
  MonitorElement* meROCNmbr_;

  static const int LINK_bits,  ROC_bits,  DCOL_bits,  PXID_bits,  ADC_bits, DataBit_bits, TRLRBGN_bits, EVTLGT_bits, TRLREND_bits;
  static const int LINK_shift, ROC_shift, DCOL_shift, PXID_shift, ADC_shift, DB0_shift, DB1_shift, DB2_shift, DB3_shift, DB4_shift, DB5_shift, DB6_shift, DB7_shift, TRLRBGN_shift, EVTLGT_shift, TRLREND_shift;
  static const uint32_t LINK_mask, ROC_mask, DCOL_mask, PXID_mask, ADC_mask, DataBit_mask;
  static const long long TRLRBGN_mask, EVTLGT_mask, TRLREND_mask;
  
};
#endif
