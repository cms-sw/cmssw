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
// $Id: HcalDbRecord.h,v 1.2 2005/10/04 18:05:54 fedor Exp $
//
#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class HcalPedestalsRcd;
class HcalPedestalWidthsRcd;
class HcalGainsRcd;
class HcalGainWidthsRcd;
class HcalQIEShapeRcd;
class HcalQIEDataRcd;
class HcalChannelQualityRcd;
class HcalElectronicsMapRcd;


// class HcalDbRecord : public edm::eventsetup::EventSetupRecordImplementation<HcalDbRecord> {};

class HcalDbRecord : public edm::eventsetup::DependentRecordImplementation <HcalDbRecord,  
  boost::mpl::vector<HcalPedestalsRcd, HcalPedestalWidthsRcd, HcalGainsRcd, HcalGainWidthsRcd, HcalQIEShapeRcd, HcalQIEDataRcd, HcalChannelQualityRcd, HcalElectronicsMapRcd > > {}; 

#endif /* HCALDBPRODUCER_HCALDBRECORD_H */

