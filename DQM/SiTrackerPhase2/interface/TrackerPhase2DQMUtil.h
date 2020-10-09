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
  MonitorElement* book1DFromPSet(const edm::ParameterSet& hpars, const std::string& hname, DQMStore::IBooker& ibooker) {
    MonitorElement* temp = nullptr;
    if (hpars.getParameter<bool>("switch")) {
      temp = ibooker.book1D(hpars.getParameter<std::string>("name"),
                            hpars.getParameter<std::string>("title"),
                            hpars.getParameter<int32_t>("NxBins"),
                            hpars.getParameter<double>("xmin"),
                            hpars.getParameter<double>("xmax"));
    }
    return temp;
  }

  MonitorElement* book2DFromPSet(const edm::ParameterSet& hpars, const std::string& hname, DQMStore::IBooker& ibooker) {
    MonitorElement* temp = nullptr;
    if (hpars.getParameter<bool>("switch")) {
      temp = ibooker.book2D(hpars.getParameter<std::string>("name"),
                            hpars.getParameter<std::string>("title"),
                            hpars.getParameter<int32_t>("NxBins"),
                            hpars.getParameter<double>("xmin"),
                            hpars.getParameter<double>("xmax"),
                            hpars.getParameter<int32_t>("NyBins"),
                            hpars.getParameter<double>("ymin"),
                            hpars.getParameter<double>("ymax"));
    }
    return temp;
  }

  MonitorElement* bookProfile1DFromPSet(const edm::ParameterSet& hpars,
                                        const std::string& hname,
                                        DQMStore::IBooker& ibooker) {
    MonitorElement* temp = nullptr;
    if (hpars.getParameter<bool>("switch")) {
      temp = ibooker.bookProfile(hpars.getParameter<std::string>("name"),
                                 hpars.getParameter<std::string>("title"),
                                 hpars.getParameter<int32_t>("NxBins"),
                                 hpars.getParameter<double>("xmin"),
                                 hpars.getParameter<double>("xmax"),
                                 hpars.getParameter<double>("ymin"),
                                 hpars.getParameter<double>("ymax"));
    }
    return temp;
  }
}  // namespace phase2tkutil
#endif
