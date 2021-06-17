#ifndef _DQM_SiTrackerPhase2_Phase2TrackerHarvestingUtil_h
#define _DQM_SiTrackerPhase2_Phase2TrackerHarvestingUtil_h
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>

namespace phase2tkharvestutil {

  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  MonitorElement* book1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker);

  MonitorElement* book2DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker);

  MonitorElement* bookProfile1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker);
}  // namespace phase2tkharvestutil
#endif
