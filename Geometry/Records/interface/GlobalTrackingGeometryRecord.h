#ifndef Records_GlobalTrackingGeometryRecord_h
#define Records_GlobalTrackingGeometryRecord_h

/** \class MuonGeometryRecord
 *  The Muon DetUnit geometry.
 *
 *  $Date: 2006/05/10 12:57:36 $
 *  $Revision: 1.1 $
 *  \author Matteo Sani
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "boost/mpl/vector.hpp"


class GlobalTrackingGeometryRecord : public edm::eventsetup::DependentRecordImplementation<GlobalTrackingGeometryRecord,boost::mpl::vector<TrackerDigiGeometryRecord,MuonGeometryRecord> > {};

#endif

