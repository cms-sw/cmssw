#ifndef HCALDBPRODUCER_HCALDBRECORD_H
#define HCALDBPRODUCER_HCALDBRECORD_H
// -*- C++ -*-
//
// Package:     HcalDbProducer
// Class  :     HcalDbRecord
// 
/**\class HcalDbRecord HcalDbRecord.h CalibFormats/HcalDbProducer/interface/HcalDbRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Tue Aug  9 19:10:36 CDT 2005
// $Id: HcalDbRecord.h,v 1.7 2008/11/08 21:16:39 rofierzy Exp $
//
#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"      
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h" 
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalL1TriggerObjectsRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"

// class HcalDbRecord : public edm::eventsetup::EventSetupRecordImplementation<HcalDbRecord> {};

class HcalDbRecord : public edm::eventsetup::DependentRecordImplementation <HcalDbRecord,  
  boost::mpl::vector<HcalPedestalsRcd, HcalPedestalWidthsRcd, HcalGainsRcd, HcalGainWidthsRcd, 
  HcalQIEDataRcd, HcalChannelQualityRcd, HcalZSThresholdsRcd, HcalRespCorrsRcd, 
  HcalL1TriggerObjectsRcd, HcalElectronicsMapRcd, HcalTimeCorrsRcd > > {}; 

#endif /* HCALDBPRODUCER_HCALDBRECORD_H */

