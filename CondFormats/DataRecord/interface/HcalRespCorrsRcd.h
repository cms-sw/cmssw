#ifndef DataRecord_HcalRespCorrsRcd_h
#define DataRecord_HcalRespCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalRespCorrsRcd
//
/**\class HcalRespCorrsRcd HcalRespCorrsRcd.h CondFormats/DataRecord/interface/HcalRespCorrsRcd.h

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
#include "CondFormats/DataRecord/interface/HBHEDarkeningRecord.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"

class HcalRespCorrsRcd
    : public edm::eventsetup::DependentRecordImplementation<
          HcalRespCorrsRcd,
          edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord, HBHEDarkeningRecord, HcalTimeSlewRecord> > {};

#endif
