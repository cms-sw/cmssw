#ifndef DataRecord_HcalCalibrationQIEDataRcd_h
#define DataRecord_HcalCalibrationQIEDataRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalCalibrationQIEDataRcd
// 
/**\class HcalCalibrationQIEDataRcd HcalCalibrationQIEDataRcd.h CondFormats/DataRecord/interface/HcalCalibrationQIEDataRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Mar  1 15:49:07 CET 2008
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalCalibrationQIEDataRcd : public edm::eventsetup::DependentRecordImplementation<HcalCalibrationQIEDataRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};

#endif
