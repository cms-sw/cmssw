#ifndef RECORDS_HCALGEOMETRYRECORD_H
#define RECORDS_HCALGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     HcalGeometryRecord
//
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentErrorExtendedRcd.h"
#include "Geometry/Records/interface/PHcalRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include <boost/mp11/list.hpp>

class HcalGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<HcalGeometryRecord,
                                                            boost::mp11::mp_list<IdealGeometryRecord,
                                                                                 HcalParametersRcd,
                                                                                 HcalSimNumberingRecord,
                                                                                 HcalRecNumberingRecord,
                                                                                 HcalAlignmentRcd,
                                                                                 HcalAlignmentErrorRcd,
                                                                                 HcalAlignmentErrorExtendedRcd,
                                                                                 GlobalPositionRcd,
                                                                                 PHcalRcd> > {};

#endif /* RECORDS_HCALGEOMETRYRECORD_H */
