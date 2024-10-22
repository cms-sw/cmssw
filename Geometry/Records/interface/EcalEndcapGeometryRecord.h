#ifndef RECORDS_ECALENDCAPGEOMETRYRECORD_H
#define RECORDS_ECALENDCAPGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     EcalEndcapGeometryRecord
//
//
// Author:      Brian Heltsley
// Created:     Tue April 1, 2008
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PEcalEndcapRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class EcalEndcapGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<EcalEndcapGeometryRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord,
                                                                             EEAlignmentRcd,
                                                                             EEAlignmentErrorRcd,
                                                                             EEAlignmentErrorExtendedRcd,
                                                                             GlobalPositionRcd,
                                                                             PEcalEndcapRcd> > {};

#endif /* RECORDS_ECALENDCAPGEOMETRYRECORD_H */
