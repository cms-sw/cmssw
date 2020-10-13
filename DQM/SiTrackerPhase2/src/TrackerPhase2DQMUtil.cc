#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
std::string phase2tkutil::getITHistoId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Disc;
  std::ostringstream fname1;
  int layer = tTopo->getITPixelLayerNumber(det_id);

  if (layer < 0)
    return "";
  if (layer < 100) {
    fname1 << "Barrel/";
    fname1 << "Layer" << layer;
    fname1 << "";
  } else {
    int side = tTopo->pxfSide(det_id);
    fname1 << "EndCap_Side" << side << "/";
    int disc = tTopo->pxfDisk(det_id);
    Disc = (disc < 9) ? "EPix" : "FPix";
    fname1 << Disc << "/";
    int ring = tTopo->pxfBlade(det_id);
    fname1 << "Ring" << ring;
  }
  return fname1.str();
}

std::string phase2tkutil::getOTHistoId(uint32_t det_id, const TrackerTopology* tTopo) {
  std::string Disc;
  std::ostringstream fname1;
  int layer = tTopo->getOTLayerNumber(det_id);

  if (layer < 0)
    return "";
  if (layer < 100) {
    fname1 << "Barrel/";
    fname1 << "Layer" << layer;
    fname1 << "";
  } else {
    int side = tTopo->tidSide(det_id);
    fname1 << "EndCap_Side" << side << "/";
    int disc = tTopo->tidWheel(det_id);
    Disc = (disc < 3) ? "TEDD_1" : "TEDD_2";
    fname1 << Disc << "/";
    int ring = tTopo->tidRing(det_id);
    fname1 << "Ring" << ring;
  }
  return fname1.str();
}
