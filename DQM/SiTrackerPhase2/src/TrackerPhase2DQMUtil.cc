#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

std::string phase2tkutil::getITHistoId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Side, Shell, Disc;
  std::ostringstream fname1;
  int layer = tTopo->getITPixelLayerNumber(det_id);
  if (layer < 0)
    return "";
  if (layer < 100) {
    fname1 << "Barrel/";
    Shell = getITShell(det_id, tTopo);
    if (Shell.empty())
      return "";
    fname1 << Shell << "/";
    fname1 << "Layer" << layer;
    fname1 << "";
  } else {
    int disc = tTopo->pxfDisk(det_id);
    Disc = (disc < 9) ? "ForwardPix" : "EndcapPix";
    Shell = getITShell(det_id, tTopo);
    fname1 << "/Endcaps/" << Disc << "/" << Shell << "/";
    int ring = tTopo->pxfBlade(det_id);
    fname1 << "Ring" << ring;
  }
  return fname1.str();
}
std::string phase2tkutil::getITHistoWheelId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Side, Shell, Disc;
  std::ostringstream fname1;
  int layer = tTopo->getITPixelLayerNumber(det_id);

  if (layer < 100) {  //This should ALWAYS be an endcap or forward histo
    return "";
  } else {
    int disc = tTopo->pxfDisk(det_id);
    Disc = (disc < 9) ? "ForwardPix" : "EndcapPix";
    Shell = getITShell(det_id, tTopo);
    fname1 << "/Endcaps/" << Disc << "/" << Shell << "/" << "Wheel" << disc;
  }
  return fname1.str();
}

std::string phase2tkutil::getOTHistoId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Disc, Side;
  std::ostringstream fname1;
  int layer = tTopo->getOTLayerNumber(det_id);

  if (layer < 0)
    return "";
  if (layer < 100) {
    fname1 << "Barrel/";
    fname1 << "Layer" << layer;
    fname1 << "";
  } else {
    fname1 << "EndCaps/";
    int side = tTopo->tidSide(det_id);
    Side = (side == 1) ? "MINUS" : "PLUS";
    fname1 << Side << "/";
    int disc = tTopo->tidWheel(det_id);
    Disc = (disc < 3) ? "TEDD_1" : "TEDD_2";
    fname1 << Disc << "/";
    int ring = tTopo->tidRing(det_id);
    fname1 << "Ring" << ring;
  }
  return fname1.str();
}

std::string phase2tkutil::getOTHistoWheelId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Disc, Side;
  std::ostringstream fname1;
  int layer = tTopo->getOTLayerNumber(det_id);

  if (layer < 100) {  //This should ALWAYS be an endcap histo
    return "";
  } else {
    fname1 << "EndCaps/";
    int side = tTopo->tidSide(det_id);
    Side = (side == 1) ? "MINUS" : "PLUS";
    fname1 << Side << "/";
    int disc = tTopo->tidWheel(det_id);
    Disc = (disc < 3) ? "TEDD_1" : "TEDD_2";
    fname1 << Disc << "/";
    fname1 << "Wheel" << disc;
  }
  return fname1.str();
}

std::string phase2tkutil::getITShell(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Side, Inner;
  std::ostringstream shellname;
  // Replace these vectors with a calculation func
  std::vector<unsigned int> nLaddersPerLayer = {12, 24, 20, 28};
  std::vector<unsigned int> nModulesPerFPixRing = {20, 32, 24, 32};
  std::vector<unsigned int> nModulesPerEPixRing = {20, 28, 36, 44, 48};
  int layer = tTopo->getITPixelLayerNumber(det_id);
  int ring = tTopo->pxfBlade(det_id);
  int disc = tTopo->pxfDisk(det_id);
  if (layer < 100) {  // Barrel
    if (layer % 2 == 0)
      Side = (tTopo->module(det_id) <= 5) ? "m" : "p";
    else
      Side = (tTopo->module(det_id) <= 4) ? "m" : "p";
    Inner = ((tTopo->pxbLadder(det_id) - (nLaddersPerLayer[layer - 1] / 4) - 1) >= (nLaddersPerLayer[layer - 1] / 2))
                ? "I"
                : "O";
    shellname << Side << Inner;
  } else if (disc < 9) {  // Forward
    int side = tTopo->tidSide(det_id);
    Side = (side == 1) ? "m" : "p";
    Inner =
        ((tTopo->pxfModule(det_id) - (nModulesPerFPixRing[ring - 1] / 4) - 1) >= (nModulesPerFPixRing[ring - 1] / 2))
            ? "I"
            : "O";
    shellname << Side << Inner;
  } else {  // Endcap
    int side = tTopo->tidSide(det_id);
    Side = (side == 1) ? "m" : "p";
    Inner =
        ((tTopo->pxfModule(det_id) - (nModulesPerEPixRing[ring - 1] / 4) - 1) >= (nModulesPerEPixRing[ring - 1] / 2))
            ? "I"
            : "O";
    shellname << Side << Inner;
  }
  return shellname.str();
}

typedef dqm::reco::MonitorElement MonitorElement;
typedef dqm::reco::DQMStore DQMStore;
MonitorElement* phase2tkutil::book1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker) {
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

MonitorElement* phase2tkutil::book2DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker) {
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

MonitorElement* phase2tkutil::bookProfile1DFromPSet(const edm::ParameterSet& hpars, DQMStore::IBooker& ibooker) {
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

void phase2tkutil::add1DDesc(edm::ParameterSetDescription& desc,
                             const std::string& psetKey,
                             const std::string& histName,
                             const std::string& xlabel,
                             const std::string& ylabel,
                             int nbins,
                             double xmin,
                             double xmax) {
  edm::ParameterSetDescription ps;
  ps.add<bool>("switch", true);
  ps.add<std::string>("name", histName);
  ps.add<std::string>("title", histName + ";" + xlabel + ";" + ylabel);
  ps.add<int>("NxBins", nbins);
  ps.add<double>("xmin", xmin);
  ps.add<double>("xmax", xmax);
  desc.add<edm::ParameterSetDescription>(psetKey, ps);
}

void phase2tkutil::add2DDesc(edm::ParameterSetDescription& desc,
                             const std::string& psetKey,
                             const std::string& histName,
                             const std::string& xlabel,
                             const std::string& ylabel,
                             int nbx,
                             double xmin,
                             double xmax,
                             int nby,
                             double ymin,
                             double ymax) {
  edm::ParameterSetDescription ps;
  ps.add<bool>("switch", true);
  ps.add<std::string>("name", histName);
  ps.add<std::string>("title", histName + ";" + xlabel + ";" + ylabel);
  ps.add<int>("NxBins", nbx);
  ps.add<double>("xmin", xmin);
  ps.add<double>("xmax", xmax);
  ps.add<int>("NyBins", nby);
  ps.add<double>("ymin", ymin);
  ps.add<double>("ymax", ymax);
  desc.add<edm::ParameterSetDescription>(psetKey, ps);
}
