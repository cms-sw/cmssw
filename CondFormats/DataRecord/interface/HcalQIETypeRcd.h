#ifndef HcalQIETypeRcd_H
#define HcalQIETypeRcd_H
// -*- C++ -*-
//
// Package:     QIEType
// Class  :     HcalQIETypeRcd
// 
/**\class HcalQIETypeRcd HcalQIETypeRcd.h CondFormats/DataRecord/interface/HcalQIETypeRcd.h

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

class HcalQIETypeRcd : public edm::eventsetup::DependentRecordImplementation<HcalQIETypeRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
