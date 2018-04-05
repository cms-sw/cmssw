#ifndef DataRecord_HcalTimeCorrsRcd_h
#define DataRecord_HcalTimeCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalTimeCorrsRcd
// 
/**\class HcalTimeCorrsRcd HcalTimeCorrsRcd.h CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h

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

class HcalTimeCorrsRcd : public edm::eventsetup::DependentRecordImplementation<HcalTimeCorrsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
