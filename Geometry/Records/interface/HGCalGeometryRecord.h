#ifndef RECORDS_HGCALGEOMETRYRECORD_H
#define RECORDS_HGCALGEOMETRYRECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PHGCalRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class HGCalGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                                HGCalGeometryRecord,
                                edm::mpl::Vector<IdealGeometryRecord, GlobalPositionRcd, PHGCalRcd> > {};

#endif /* RECORDS_HGCALGEOMETRYRECORD_H */
