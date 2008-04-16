#ifndef SiPixelMonitorDigi_SiPixelDigiModule_h
#define SiPixelMonitorDigi_SiPixelDigiModule_h
// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelDigiModule
// 
/**\class 

 Description: Digi monitoring elements for a Pixel sensor

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  
// $Id: SiPixelDigiModule.h,v 1.6 2007/04/04 13:57:04 chiochia Exp $
//
//
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/cstdint.hpp>

class SiPixelDigiModule {        

 public:

  /// Default constructor
  SiPixelDigiModule();
  /// Constructor with raw DetId
  SiPixelDigiModule(const uint32_t& id);
  /// Constructor with raw DetId and sensor size
  SiPixelDigiModule(const uint32_t& id, const int& ncols, const int& nrows);
  /// Destructor
  ~SiPixelDigiModule();

  typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig);
  void bookUpperLevelMEs(DQMStore* theDMBE);
  void bookUpperLevelBarrelMEs(DQMStore* theDMBE);
  void bookUpperLevelEndcapMEs(DQMStore* theDMBE);
  /// Fill histograms
  void fill(bool modon, const edm::DetSetVector<PixelDigi> & input);
  
 private:

  uint32_t id_;
  int ncols_;
  int nrows_;
  MonitorElement* meNDigis_;
  MonitorElement* meADC_;
  MonitorElement* mePixDigis_;
  
    MonitorElement* Barrel_SmIL1_ndigis;
    MonitorElement* Barrel_SmIL1_adc;
    MonitorElement* Barrel_SmIL1_hitmap;
    MonitorElement* Barrel_SmOL1_ndigis;
    MonitorElement* Barrel_SmOL1_adc;
    MonitorElement* Barrel_SmOL1_hitmap;
    MonitorElement* Barrel_SpIL1_ndigis;
    MonitorElement* Barrel_SpIL1_adc;
    MonitorElement* Barrel_SpIL1_hitmap;
    MonitorElement* Barrel_SpOL1_ndigis;
    MonitorElement* Barrel_SpOL1_adc;
    MonitorElement* Barrel_SpOL1_hitmap;
    MonitorElement* Barrel_SmIL2_ndigis;
    MonitorElement* Barrel_SmIL2_adc;
    MonitorElement* Barrel_SmIL2_hitmap;
    MonitorElement* Barrel_SmOL2_ndigis;
    MonitorElement* Barrel_SmOL2_adc;
    MonitorElement* Barrel_SmOL2_hitmap;
    MonitorElement* Barrel_SpIL2_ndigis;
    MonitorElement* Barrel_SpIL2_adc;
    MonitorElement* Barrel_SpIL2_hitmap;
    MonitorElement* Barrel_SpOL2_ndigis;
    MonitorElement* Barrel_SpOL2_adc;
    MonitorElement* Barrel_SpOL2_hitmap;
    MonitorElement* Barrel_SmIL3_ndigis;
    MonitorElement* Barrel_SmIL3_adc;
    MonitorElement* Barrel_SmIL3_hitmap;
    MonitorElement* Barrel_SmOL3_ndigis;
    MonitorElement* Barrel_SmOL3_adc;
    MonitorElement* Barrel_SmOL3_hitmap;
    MonitorElement* Barrel_SpIL3_ndigis;
    MonitorElement* Barrel_SpIL3_adc;
    MonitorElement* Barrel_SpIL3_hitmap;
    MonitorElement* Barrel_SpOL3_ndigis;
    MonitorElement* Barrel_SpOL3_adc;
    MonitorElement* Barrel_SpOL3_hitmap;
    MonitorElement* Endcap_HCmID1_ndigis;
    MonitorElement* Endcap_HCmID1_adc;	 
    MonitorElement* Endcap_HCmID1_hitmap;
    MonitorElement* Endcap_HCmOD1_ndigis;
    MonitorElement* Endcap_HCmOD1_adc;
    MonitorElement* Endcap_HCmOD1_hitmap;
    MonitorElement* Endcap_HCpID1_ndigis;
    MonitorElement* Endcap_HCpID1_adc;
    MonitorElement* Endcap_HCpID1_hitmap;
    MonitorElement* Endcap_HCpOD1_ndigis;
    MonitorElement* Endcap_HCpOD1_adc;
    MonitorElement* Endcap_HCpOD1_hitmap;
    MonitorElement* Endcap_HCmID2_ndigis;
    MonitorElement* Endcap_HCmID2_adc;
    MonitorElement* Endcap_HCmID2_hitmap;
    MonitorElement* Endcap_HCmOD2_ndigis;
    MonitorElement* Endcap_HCmOD2_adc;
    MonitorElement* Endcap_HCmOD2_hitmap;
    MonitorElement* Endcap_HCpID2_ndigis;
    MonitorElement* Endcap_HCpID2_adc;
    MonitorElement* Endcap_HCpID2_hitmap;
    MonitorElement* Endcap_HCpOD2_ndigis;
    MonitorElement* Endcap_HCpOD2_adc;
    MonitorElement* Endcap_HCpOD2_hitmap; 
    
    bool bookedUL_;
};
#endif
