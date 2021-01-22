#ifndef _DQM_SiTrackerPhase2_Phase2TrackerValidationUtil_h
#define _DQM_SiTrackerPhase2_Phase2TrackerValidationUtil_h
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include <sstream>

namespace phase2tkutil {

  std::string getITHistoId(uint32_t det_id, const TrackerTopology* tTopo);
  std::string getOTHistoId(uint32_t det_id, const TrackerTopology* tTopo);

  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  MonitorElement* book1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker);

  MonitorElement* book2DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker);

  MonitorElement* bookProfile1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker);
}  // namespace phase2tkutil
#endif
