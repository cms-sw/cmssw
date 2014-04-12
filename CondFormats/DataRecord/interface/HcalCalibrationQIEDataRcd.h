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
// $Id: HcalCalibrationQIEDataRcd.h,v 1.1 2008/03/03 16:56:47 rofierzy Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalCalibrationQIEDataRcd : public edm::eventsetup::DependentRecordImplementation<HcalCalibrationQIEDataRcd, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
