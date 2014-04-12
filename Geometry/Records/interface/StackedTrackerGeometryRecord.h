
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
#include "boost/mpl/vector.hpp"

class StackedTrackerGeometryRecord : public edm::eventsetup::DependentRecordImplementation< StackedTrackerGeometryRecord , boost::mpl::vector<TrackerDigiGeometryRecord> > {};

#endif 

/* RECORDS_StackedTrackerGEOMETRYRECORD_H */


