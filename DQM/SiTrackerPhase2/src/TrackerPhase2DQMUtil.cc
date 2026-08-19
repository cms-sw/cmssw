#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

// Unified folder getter for IT and OT
// Gets the geographical information in either filepath or "pretty" format
// Uses the LEVEL to figure out which information to include
// LEVEL == 1: InnerTracker or OuterTracker (if LEVEL == 0, this is the behaviour)
// LEVEL == 2: Barrel or Endcap or Forward(IT)
// LEVEL == 3: Barrel or Endcap Shells (IT), Endcap Sides (OT)
// LEVEL == 4: Endcap rings
// LEVEL == 5: Endcap wheels
// LEVEL == 6: Barrel layers or endcap rings in wheels
std::string phase2tkutil::getHistoId(uint32_t det_id, const TrackerTopology* tTopo, float phi, int LEVEL, bool pretty) {
  std::ostringstream foldername;
  std::string Substructure, Side, Shell, TEDD;
  int layer = -1, wheel = -1, ring = -1;
  bool inner = (DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel ||
                DetId(det_id).subdetId() == PixelSubdetector::PixelEndcap);

  if (DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel ||
      DetId(det_id).subdetId() == SiStripSubdetector::TOB) {
    Substructure = (pretty ? "Barrel " : "Barrel/");
    if (inner)
      layer = tTopo->getITPixelLayerNumber(det_id);
    else
      layer = tTopo->getOTLayerNumber(det_id);
  } else if (DetId(det_id).subdetId() == PixelSubdetector::PixelEndcap ||
             DetId(det_id).subdetId() == SiStripSubdetector::TID) {
    Substructure = (pretty ? "endcap " : "Endcaps/");
    if (inner) {
      wheel = tTopo->pxfDisk(det_id);
      ring = tTopo->pxfBlade(det_id);

      if (wheel < 9)
        Substructure.append(pretty ? "FPix " : "ForwardPix/");
      else
        Substructure.append(pretty ? "EPix " : "EndcapPix/");

    } else {
      int side = tTopo->tidSide(det_id);
      Side = (pretty ? ((side == 1) ? "side minus " : "side plus ") : ((side == 1) ? "MINUS/" : "PLUS/"));
      wheel = tTopo->tidWheel(det_id);
      TEDD = (pretty ? (wheel < 3 ? "TEDD_1 " : "TEDD_2 ") : ((wheel < 3) ? "TEDD_1/" : "TEDD_2/"));
      ring = tTopo->tidRing(det_id);
    }
  } else {  //unknown subdetector - should probably throw
    return "ERROR";
  }

  if (inner) {
    foldername << (pretty ? "IT " : "");
    Shell = getITShell(det_id, tTopo, phi);
  } else {
    foldername << (pretty ? "OT " : "");
  }

  if (LEVEL > 1)
    foldername << Substructure;

  if (LEVEL > 2) {
    if (inner)
      foldername << (pretty ? "shell " : "") << Shell << (pretty ? " " : "/");
    else if (DetId(det_id).subdetId() == SiStripSubdetector::TID)
      foldername << Side;
  }

  if (LEVEL == 4) {
    if (DetId(det_id).subdetId() == SiStripSubdetector::TID)
      foldername << TEDD << "Ring" << ring << (pretty ? " " : "/");
    else if (DetId(det_id).subdetId() == PixelSubdetector::PixelEndcap)
      foldername << "Ring" << ring << (pretty ? " " : "/");
  }

  if (LEVEL > 4) {
    if (DetId(det_id).subdetId() == PixelSubdetector::PixelEndcap)
      foldername << "Wheel" << wheel << (pretty ? " " : "/");
    else if (DetId(det_id).subdetId() == SiStripSubdetector::TID)
      foldername << TEDD << "Wheel" << wheel << (pretty ? " " : "/");
  }

  if (LEVEL > 5) {
    if (DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel ||
        DetId(det_id).subdetId() == SiStripSubdetector::TOB)
      foldername << "Layer" << layer << (pretty ? " " : "/");
    else
      foldername << "Ring" << ring << (pretty ? " " : "/");
  }
  return foldername.str();
}

std::string phase2tkutil::getITShell(uint32_t det_id, const TrackerTopology* tTopo, float phi) {
  std::string Side, Inner;
  std::ostringstream shellname;
  int layer = tTopo->getITPixelLayerNumber(det_id);
  if (DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel) {
    if (layer % 2 == 0)
      Side = (tTopo->module(det_id) <= 5) ? "m" : "p";
    else
      Side = (tTopo->module(det_id) <= 4) ? "m" : "p";
  } else {
    int side = tTopo->tidSide(det_id);
    Side = (side == 1) ? "m" : "p";
  }
  Inner = (std::abs(phi) > 3.1415 / 2 ? "O" : "I");
  shellname << Side << Inner;
  return shellname.str();
}

int phase2tkutil::getITSignedModule(uint32_t det_id, const TrackerTopology* tTopo, float phi) {
  int signedModule;
  int module = tTopo->module(det_id);
  int layer = tTopo->getITPixelLayerNumber(det_id);
  if (layer % 2 == 0)
    signedModule = (module <= 5 ? module - 6 : module - 5);
  else
    signedModule = (module <= 4 ? module - 5 : module - 4);

  return signedModule;
}

int phase2tkutil::getITSignedLadder(uint32_t det_id, const TrackerTopology* tTopo, float phi) {
  int signedLadder;
  int ladder = tTopo->pxbLadder(det_id);
  int layer = tTopo->getITPixelLayerNumber(det_id);
  if (std::abs(phi) > 3.1415 / 2) {  // Outer shell
    if (layer == 1)
      signedLadder = ladder - 10;
    if (layer == 2)
      signedLadder = ladder - 19;
    if (layer == 3)
      signedLadder = ladder - 16;
    if (layer == 4)
      signedLadder = ladder - 22;
  } else {  // Inner shell
    if (layer == 1)
      signedLadder = (ladder > 9 ? ladder - 9 : ladder + 3);
    if (layer == 2)
      signedLadder = (ladder > 18 ? ladder - 18 : ladder + 6);
    if (layer == 3)
      signedLadder = (ladder > 15 ? ladder - 15 : ladder + 5);
    if (layer == 4)
      signedLadder = (ladder > 21 ? ladder - 21 : ladder + 7);
  }

  return signedLadder;
}

typedef dqm::reco::MonitorElement MonitorElement;
typedef dqm::reco::DQMStore DQMStore;
MonitorElement* phase2tkutil::book1DFromPSet(const edm::ParameterSet& hpars,
                                             DQMStore::IBooker& ibooker,
                                             std::string titleString,
                                             int scale) {
  MonitorElement* temp = nullptr;
  if (hpars.getParameter<bool>("switch")) {
    double xMax = hpars.getParameter<double>("xmax");
    std::string title = hpars.getParameter<std::string>("title");
    xMax = xMax / scale;
    if (!titleString.empty())
      title = std::vformat(title, std::make_format_args(titleString));
    temp = ibooker.book1D(hpars.getParameter<std::string>("name"),
                          title,
                          hpars.getParameter<int32_t>("NxBins"),
                          hpars.getParameter<double>("xmin"),
                          xMax);
  }
  return temp;
}

MonitorElement* phase2tkutil::book2DFromPSet(const edm::ParameterSet& hpars,
                                             DQMStore::IBooker& ibooker,
                                             std::string titleString) {
  MonitorElement* temp = nullptr;
  if (hpars.getParameter<bool>("switch")) {
    std::string title = hpars.getParameter<std::string>("title");
    if (!titleString.empty())
      title = std::vformat(title, std::make_format_args(titleString));
    temp = ibooker.book2D(hpars.getParameter<std::string>("name"),
                          title,
                          hpars.getParameter<int32_t>("NxBins"),
                          hpars.getParameter<double>("xmin"),
                          hpars.getParameter<double>("xmax"),
                          hpars.getParameter<int32_t>("NyBins"),
                          hpars.getParameter<double>("ymin"),
                          hpars.getParameter<double>("ymax"));
  }
  return temp;
}

MonitorElement* phase2tkutil::bookProfile1DFromPSet(const edm::ParameterSet& hpars,
                                                    DQMStore::IBooker& ibooker,
                                                    std::string titleString) {
  MonitorElement* temp = nullptr;
  if (hpars.getParameter<bool>("switch")) {
    std::string title = hpars.getParameter<std::string>("title");
    if (!titleString.empty())
      title = std::vformat(title, std::make_format_args(titleString));
    temp = ibooker.bookProfile(hpars.getParameter<std::string>("name"),
                               title,
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
                             const std::string& histTitle,
                             const std::string& xlabel,
                             const std::string& ylabel,
                             int nbins,
                             double xmin,
                             double xmax) {
  edm::ParameterSetDescription ps;
  ps.add<bool>("switch", true);
  ps.add<std::string>("name", histName);
  ps.add<std::string>("title", histTitle + ";" + xlabel + ";" + ylabel);
  ps.add<int>("NxBins", nbins);
  ps.add<double>("xmin", xmin);
  ps.add<double>("xmax", xmax);
  desc.add<edm::ParameterSetDescription>(psetKey, ps);
}

void phase2tkutil::add2DDesc(edm::ParameterSetDescription& desc,
                             const std::string& psetKey,
                             const std::string& histName,
                             const std::string& histTitle,
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
  ps.add<std::string>("title", histTitle + ";" + xlabel + ";" + ylabel);
  ps.add<int>("NxBins", nbx);
  ps.add<double>("xmin", xmin);
  ps.add<double>("xmax", xmax);
  ps.add<int>("NyBins", nby);
  ps.add<double>("ymin", ymin);
  ps.add<double>("ymax", ymax);
  desc.add<edm::ParameterSetDescription>(psetKey, ps);
}
