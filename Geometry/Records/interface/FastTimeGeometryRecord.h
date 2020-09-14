#ifndef Records_FastTimeGeometryRecord_h
#define Records_FastTimeGeometryRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PFastTimeRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class FastTimeGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                                   FastTimeGeometryRecord,
                                   edm::mpl::Vector<IdealGeometryRecord, GlobalPositionRcd, PFastTimeRcd> > {};

#endif /* Records_FastTimeGeometryRecord_h */
