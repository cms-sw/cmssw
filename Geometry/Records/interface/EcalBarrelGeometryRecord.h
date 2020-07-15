#ifndef RECORDS_ECALBARRELGEOMETRYRECORD_H
#define RECORDS_ECALBARRELGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     EcalBarrelGeometryRecord
//
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PEcalBarrelRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include <boost/mp11/list.hpp>

class EcalBarrelGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<EcalBarrelGeometryRecord,
                                                            boost::mp11::mp_list<IdealGeometryRecord,
                                                                               EBAlignmentRcd,
                                                                               EBAlignmentErrorRcd,
                                                                               EBAlignmentErrorExtendedRcd,
                                                                               GlobalPositionRcd,
                                                                               PEcalBarrelRcd> > {};

#endif /* RECORDS_ECALBARRELGEOMETRYRECORD_H */
