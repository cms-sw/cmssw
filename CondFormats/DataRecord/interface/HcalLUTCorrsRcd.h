#ifndef DataRecord_HcalLUTCorrsRcd_h
#define DataRecord_HcalLUTCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalLUTCorrsRcd
// 
/**\class HcalLUTCorrsRcd HcalLUTCorrsRcd.h CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Mar  1 15:49:28 CET 2008
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalLUTCorrsRcd : public edm::eventsetup::DependentRecordImplementation<HcalLUTCorrsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
