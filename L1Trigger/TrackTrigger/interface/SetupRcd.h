#ifndef L1Trigger_TrackTrigger_SetupRcd_h
#define L1Trigger_TrackTrigger_SetupRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

namespace tt {

  typedef edm::mpl::
      Vector<TrackerDigiGeometryRecord, TrackerTopologyRcd, TrackerDetToDTCELinkCablingMapRcd, TTStubAlgorithmRecord>
          SetupDepRcds;

  // record of tt::Setup
  class SetupRcd : public edm::eventsetup::DependentRecordImplementation<SetupRcd, SetupDepRcds> {};

}  // namespace tt

#endif
