#ifndef GeometryRecord_HcalRecNumberingRecord_h
#define GeometryRecord_HcalRecNumberingRecord_h
// -*- C++ -*-
//
// Package:     Record
// Class  :     HcalRecNumberingRecord
//
/**\class HcalRecNumberingRecord HcalRecNumberingRecord.h Geometry/Record/interface/HcalRecNumberingRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Thu Dec 24 16:41:02 PDT 2013
//
#include <boost/mp11/list.hpp>
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"

class HcalRecNumberingRecord
    : public edm::eventsetup::DependentRecordImplementation<
          HcalRecNumberingRecord,
          boost::mp11::mp_list<IdealGeometryRecord, HcalParametersRcd, HcalSimNumberingRecord> > {};

#endif
