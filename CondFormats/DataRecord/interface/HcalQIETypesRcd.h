#ifndef HcalQIETypesRcd_H
#define HcalQIETypesRcd_H
// -*- C++ -*-
//
// Package:     QIETypes
// Class  :     HcalQIETypesRcd
// 
/**\class HcalQIETypesRcd HcalQIETypesRcd.h CondFormats/DataRecord/interface/HcalQIETypesRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Walter Alda
// Created:     Nov  11 2015
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalQIETypesRcd : public edm::eventsetup::DependentRecordImplementation<HcalQIETypesRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
