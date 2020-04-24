#ifndef DataRecord_HcalValidationCorrsRcd_h
#define DataRecord_HcalValidationCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalValidationCorrsRcd
// 
/**\class HcalValidationCorrsRcd HcalValidationCorrsRcd.h CondFormats/DataRecord/interface/HcalValidationCorrsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Gena Kukartsev
// Created:     Wed Jul 29 14:35:28 CSET 2009
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalValidationCorrsRcd : public edm::eventsetup::DependentRecordImplementation<HcalValidationCorrsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
