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
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class TrackerDigiGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<TrackerDigiGeometryRecord,
                                                            edm::mpl::Vector<IdealGeometryRecord,
                                                                             TrackerAlignmentRcd,
                                                                             TrackerAlignmentErrorExtendedRcd,
                                                                             TrackerSurfaceDeformationRcd,
                                                                             GlobalPositionRcd,
                                                                             TrackerTopologyRcd,
                                                                             PTrackerParametersRcd,
                                                                             PTrackerAdditionalParametersPerDetRcd> > {
};

#endif /* RECORDS_TRACKERDIGIGEOMETRYRECORD_H */
