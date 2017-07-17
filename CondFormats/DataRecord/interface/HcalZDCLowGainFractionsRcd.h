#ifndef HcalZDCLowGainFractionsRcd_H
#define HcalZDCLowGainFractionsRcd_H
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     HcalZDCLowGainFractionsRcd
// 
/**\class HcalZDCLowGainFractionsRcd HcalZDCLowGainFractionsRcd.h CondFormats/DataRecord/interface/HcalZDCLowGainFractionsRcd.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Author:      Audrius Mecionis
// Created:     Wed, 24 Sep 2014 11:27:57 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalZDCLowGainFractionsRcd : public edm::eventsetup::DependentRecordImplementation<HcalZDCLowGainFractionsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
