#include "DQM/SiTrackerPhase2/interface/TrackerPhase2HarvestingUtil.h"
typedef dqm::harvesting::MonitorElement MonitorElement;
typedef dqm::harvesting::DQMStore DQMStore;
MonitorElement* phase2tkharvestutil::book1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker) {
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

MonitorElement* phase2tkharvestutil::book2DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker) {
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

MonitorElement* phase2tkharvestutil::bookProfile1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker) {
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
