#ifndef DataRecord_HcalPFCorrsRcd_h
#define DataRecord_HcalPFCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalPFCorrsRcd
// 
/**\class HcalPFCorrsRcd HcalPFCorrsRcd.h CondFormats/DataRecord/interface/HcalPFCorrsRcd.h

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

class HcalPFCorrsRcd : public edm::eventsetup::DependentRecordImplementation<HcalPFCorrsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
