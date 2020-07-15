#ifndef RECORDS_TRACKERDIGIGEOMETRYRECORD_H
#define RECORDS_TRACKERDIGIGEOMETRYRECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include <boost/mp11/list.hpp>

class TrackerDigiGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<TrackerDigiGeometryRecord,
                                                            boost::mp11::mp_list<IdealGeometryRecord,
                                                                               TrackerAlignmentRcd,
                                                                               TrackerAlignmentErrorExtendedRcd,
                                                                               TrackerSurfaceDeformationRcd,
                                                                               GlobalPositionRcd,
                                                                               TrackerTopologyRcd,
                                                                               PTrackerParametersRcd> > {};

#endif /* RECORDS_TRACKERDIGIGEOMETRYRECORD_H */
