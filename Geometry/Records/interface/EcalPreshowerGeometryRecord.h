#ifndef RECORDS_ECALPRESHOWERGEOMETRYRECORD_H
#define RECORDS_ECALPRESHOWERGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     EcalPreshowerGeometryRecord
//
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PEcalPreshowerRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class EcalPreshowerGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<EcalPreshowerGeometryRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord,
                                                                             ESAlignmentRcd,
                                                                             ESAlignmentErrorRcd,
                                                                             ESAlignmentErrorExtendedRcd,
                                                                             GlobalPositionRcd,
                                                                             PEcalPreshowerRcd> > {};

#endif /* RECORDS_ECALPRESHOWERGEOMETRYRECORD_H */
