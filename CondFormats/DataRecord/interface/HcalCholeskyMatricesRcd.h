#ifndef DataRecord_HcalCholeskyMatricesRcd_h
#define DataRecord_HcalCholeskyMatricesRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalCholeskyMatricesRcd
// 
/**\class HcalCholeskyMatricesRcd HcalCholeskyMatricesRcd.h CondFormats/DataRecord/interface/HcalCholeskyMatricesRcd.h

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

class HcalCholeskyMatricesRcd : public edm::eventsetup::DependentRecordImplementation<HcalCholeskyMatricesRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
