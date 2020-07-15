#ifndef Records_GlobalTrackingGeometryRecord_h
#define Records_GlobalTrackingGeometryRecord_h

/** \class MuonGeometryRecord
 *  The Muon DetUnit geometry.
 *
 *  \author Matteo Sani
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include <boost/mp11/list.hpp>

class GlobalTrackingGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<
          GlobalTrackingGeometryRecord,
          boost::mp11::mp_list<TrackerDigiGeometryRecord, MTDDigiGeometryRecord, MuonGeometryRecord> > {};

#endif
