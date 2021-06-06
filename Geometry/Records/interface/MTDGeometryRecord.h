#ifndef RECORDS_MTDGEOMETRYRECORD_H
#define RECORDS_MTDGEOMETRYRECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PFastTimeRcd.h"
#include "Geometry/Records/interface/BTLGeometryRcd.h"
#include "Geometry/Records/interface/ETLGeometryRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class MTDGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<
          MTDGeometryRecord,
          edm::mpl::Vector<IdealGeometryRecord, BTLGeometryRcd, ETLGeometryRcd, GlobalPositionRcd, PFastTimeRcd> > {};

#endif /* RECORDS_MTDGEOMETRYRECORD_H */
