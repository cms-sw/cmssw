
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_GEOMETRY_RECORD_H
#define STACKED_TRACKER_GEOMETRY_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include <boost/mp11/list.hpp>

class StackedTrackerGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<StackedTrackerGeometryRecord,
                                                            boost::mp11::mp_list<TrackerDigiGeometryRecord> > {};

#endif

/* RECORDS_StackedTrackerGEOMETRYRECORD_H */
