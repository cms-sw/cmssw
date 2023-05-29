#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/SiStripSubStructure.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

// TkLayerMap

void TkLayerMap::initMap(const std::vector<uint32_t>& TkDetIdList) {
  switch (layer_) {
    case TkLayerMap::TIB_L1:
    case TkLayerMap::TIB_L2:
    case TkLayerMap::TIB_L3:
    case TkLayerMap::TIB_L4:
      initMap_TIB(TkDetIdList);
      break;
    case TkLayerMap::TIDP_D1:
    case TkLayerMap::TIDP_D2:
    case TkLayerMap::TIDP_D3:
    case TkLayerMap::TIDM_D1:
    case TkLayerMap::TIDM_D2:
    case TkLayerMap::TIDM_D3:
      initMap_TID(TkDetIdList);
      break;
    case TkLayerMap::TOB_L1:
    case TkLayerMap::TOB_L2:
    case TkLayerMap::TOB_L3:
    case TkLayerMap::TOB_L4:
    case TkLayerMap::TOB_L5:
    case TkLayerMap::TOB_L6:
      initMap_TOB(TkDetIdList);
      break;
    case TkLayerMap::TECP_W1:
    case TkLayerMap::TECP_W2:
    case TkLayerMap::TECP_W3:
    case TkLayerMap::TECP_W4:
    case TkLayerMap::TECP_W5:
    case TkLayerMap::TECP_W6:
    case TkLayerMap::TECP_W7:
    case TkLayerMap::TECP_W8:
    case TkLayerMap::TECP_W9:
    case TkLayerMap::TECM_W1:
    case TkLayerMap::TECM_W2:
    case TkLayerMap::TECM_W3:
    case TkLayerMap::TECM_W4:
    case TkLayerMap::TECM_W5:
    case TkLayerMap::TECM_W6:
    case TkLayerMap::TECM_W7:
    case TkLayerMap::TECM_W8:
    case TkLayerMap::TECM_W9:
      initMap_TEC(TkDetIdList);
      break;
    default:
      edm::LogError("TkLayerMap") << " TkLayerMap::requested creation of a wrong layer Nb " << layer_;
  }
}

void TkLayerMap::initMap_TIB(const std::vector<uint32_t>& TkDetIdList) {
  //extract  vector of module in the layer
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure::getTIBDetectors(TkDetIdList, LayerDetIdList, tTopo_, layer_, 0, 0, 0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTIB12] layer " << layer_ << " number of dets " << LayerDetIdList.size()
                         << " lowY " << lowY_ << " high " << highY_ << " Nstring " << nStringExt_;

  for (uint32_t det : LayerDetIdList) {
    const auto xyb = getXY_TIB(det);
    binToDet_[bin(xyb.ix, xyb.iy)] = det;

    LogTrace("TkLayerMap") << "[TkLayerMap::createTIB] " << det << " " << xyb.ix << " " << xyb.iy << " " << xyb.x << " "
                           << xyb.y;
  }
}

void TkLayerMap::initMap_TOB(const std::vector<uint32_t>& TkDetIdList) {
  //extract  vector of module in the layer
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure::getTOBDetectors(TkDetIdList, LayerDetIdList, tTopo_, layer_ - 10, 0, 0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] layer " << layer_ - 10 << " number of dets "
                         << LayerDetIdList.size() << " lowY " << lowY_ << " high " << highY_ << " Nstring "
                         << nStringExt_;

  for (uint32_t det : LayerDetIdList) {
    const auto xyb = getXY_TOB(det);
    binToDet_[bin(xyb.ix, xyb.iy)] = det;

    LogTrace("TkLayerMap") << "[TkLayerMap::createTOB] " << det << " " << xyb.ix << " " << xyb.iy << " " << xyb.x << " "
                           << xyb.y;
  }
}

void TkLayerMap::initMap_TID(const std::vector<uint32_t>& TkDetIdList) {
  //extract  vector of module in the layer
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure::getTIDDetectors(TkDetIdList,
                                       LayerDetIdList,
                                       tTopo_,
                                       (layer_ - TkLayerMap::TIDM_D1) / 3 + 1,
                                       (layer_ - TkLayerMap::TIDM_D1) % 3 + 1,
                                       0,
                                       0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTID] layer side " << (layer_ - TkLayerMap::TIDM_D1) / 3 + 1 << " nb "
                         << (layer_ - TkLayerMap::TIDM_D1) % 3 + 1 << " number of dets " << LayerDetIdList.size()
                         << " lowY " << lowY_ << " high " << highY_ << " Nstring " << nStringExt_;

  for (uint32_t det : LayerDetIdList) {
    const auto xyb = getXY_TID(det);
    binToDet_[bin(xyb.ix, xyb.iy)] = det;

    LogTrace("TkLayerMap") << "[TkLayerMap::createTID] " << det << " " << xyb.ix << " " << xyb.iy << " " << xyb.x << " "
                           << xyb.y;
  }
}

void TkLayerMap::initMap_TEC(const std::vector<uint32_t>& TkDetIdList) {
  //extract  vector of module in the layer
  std::vector<uint32_t> LayerDetIdList;
  SiStripSubStructure::getTECDetectors(TkDetIdList,
                                       LayerDetIdList,
                                       tTopo_,
                                       (layer_ - TkLayerMap::TECM_W1) / 9 + 1,
                                       (layer_ - TkLayerMap::TECM_W1) % 9 + 1,
                                       0,
                                       0);

  LogTrace("TkLayerMap") << "[TkLayerMap::createTEC] layer side " << (layer_ - TkLayerMap::TECM_W1) / 9 + 1 << " "
                         << (layer_ - TkLayerMap::TECM_W1) % 9 + 1 << " number of dets " << LayerDetIdList.size()
                         << " lowY " << lowY_ << " high " << highY_ << " Nstring " << nStringExt_;

  for (uint32_t det : LayerDetIdList) {
    const auto xyb = getXY_TEC(det);
    binToDet_[bin(xyb.ix, xyb.iy)] = det;

    LogTrace("TkLayerMap") << "[TkLayerMap::createTEC] " << det << " " << xyb.ix << " " << xyb.iy << " " << xyb.x << " "
                           << xyb.y;
  }
}

const int16_t TkLayerMap::layerSearch(DetId detid, const TrackerTopology* tTopo) {
  switch (detid.subdetId()) {
    case SiStripDetId::TIB:
      return TkLayerMap::TIB_L1 - 1 + tTopo->tibLayer(detid);
    case SiStripDetId::TID:
      return TkLayerMap::TIDM_D1 - 1 + (tTopo->tidSide(detid) - 1) * 3 + tTopo->tidWheel(detid);
    case SiStripDetId::TOB:
      return TkLayerMap::TOB_L1 - 1 + tTopo->tobLayer(detid);
    case SiStripDetId::TEC:
      return TkLayerMap::TECM_W1 - 1 + (tTopo->tecSide(detid) - 1) * 9 + tTopo->tecWheel(detid);
    default:
      return TkLayerMap::INVALID;
  }
}

const TkLayerMap::XYbin TkLayerMap::getXY(DetId detid, int layerEnumNb) const {
  LogTrace("TkLayerMap") << "[TkLayerMap::getXY] " << detid.rawId() << " layer " << layerEnumNb;

  if (!layerEnumNb)
    layerEnumNb = layerSearch(detid, tTopo_);

  if (layerEnumNb != layer_)
    throw cms::Exception("CorruptedData")
        << "[TkLayerMap::getXY] Fill of DetId " << detid.rawId() << " layerEnumNb " << layerEnumNb
        << " are requested to wrong TkLayerMap " << layer_ << " \nPlease check the TkDetMap code";

  if (layerEnumNb >= TkLayerMap::TIB_L1 && layerEnumNb <= TkLayerMap::TIB_L4)
    return getXY_TIB(detid);
  else if (layerEnumNb >= TkLayerMap::TIDM_D1 && layerEnumNb <= TkLayerMap::TIDP_D3)
    return getXY_TID(detid);
  else if (layerEnumNb >= TkLayerMap::TOB_L1 && layerEnumNb <= TkLayerMap::TOB_L6)
    return getXY_TOB(detid);
  else
    return getXY_TEC(detid);
}

TkLayerMap::XYbin TkLayerMap::getXY_TIB(DetId detid) const {
  XYbin xyb;
  xyb.ix = (2 * (tTopo_->tibIsZMinusSide(detid) ? -1 * tTopo_->tibModule(detid) + 3 : tTopo_->tibModule(detid) + 2) +
            (tTopo_->tibLayer(detid) % 2 ? (tTopo_->tibIsInternalString(detid) ? 2 : 1)
                                         : (tTopo_->tibIsInternalString(detid) ? 1 : 2)));
  xyb.iy = ((tTopo_->tibIsInternalString(detid) ? tTopo_->tibString(detid) + singleExtStr_[tTopo_->tibString(detid)]
                                                : tTopo_->tibString(detid)) +
            ((tTopo_->tibLayer(detid) < 3) && (!tTopo_->tibIsStereo(detid)) ? nStringExt_ + 2 : 0));

  xyb.x = lowX_ + xyb.ix - 0.5;
  xyb.y = lowY_ + xyb.iy - 0.5;

  return xyb;
}

TkLayerMap::XYbin TkLayerMap::getXY_TOB(DetId detid) const {
  XYbin xyb;
  xyb.ix = (tTopo_->tobIsZMinusSide(detid) ? -1 * tTopo_->tobModule(detid) + 7 : tTopo_->tobModule(detid) + 6);
  xyb.iy = (tTopo_->tobRod(detid) + (tTopo_->tobLayer(detid) < 3 && !tTopo_->tobIsStereo(detid) ? nRod_ + 2 : 0));

  xyb.x = lowX_ + xyb.ix - 0.5;
  xyb.y = lowY_ + xyb.iy - 0.5;

  return xyb;
}

TkLayerMap::XYbin TkLayerMap::getXY_TID(DetId detid) const {
  XYbin xyb;
  xyb.ix = ((tTopo_->tidIsZMinusSide(detid) ? -3 * tTopo_->tidRing(detid) + 10 : 3 * tTopo_->tidRing(detid) - 2) +
            (tTopo_->tidIsStereo(detid) ? (tTopo_->tidIsZMinusSide(detid) ? -1 : 1) : 0));
  xyb.iy = 2 * tTopo_->tidModule(detid) - (tTopo_->tidIsBackRing(detid) ? 0 : 1);

  xyb.x = lowX_ + xyb.ix - 0.5;
  xyb.y = lowY_ + xyb.iy - 0.5;

  return xyb;
}

TkLayerMap::XYbin TkLayerMap::getXY_TEC(DetId detid) const {
  XYbin xyb;
  xyb.ix = ((tTopo_->tecIsZMinusSide(detid) ? binForRing_[7] - binForRing_[tTopo_->tecRing(detid)] + 1
                                            : binForRing_[tTopo_->tecRing(detid)]) +
            (tTopo_->tecIsStereo(detid) ? (tTopo_->tecIsZMinusSide(detid) ? -1 : 1) : 0));

  xyb.iy = ((tTopo_->tecPetalNumber(detid) - 1) *
                (modulesInRingFront_[tTopo_->tecRing(detid)] + modulesInRingBack_[tTopo_->tecRing(detid)]) +
            (tTopo_->tecIsZMinusSide(detid)
                 ? modulesInRingFront_[tTopo_->tecRing(detid)] - tTopo_->tecModule(detid) + 1 +
                       (tTopo_->tecIsBackPetal(detid) ? modulesInRingBack_[tTopo_->tecRing(detid)] : 0)
                 : tTopo_->tecModule(detid) +
                       (tTopo_->tecIsBackPetal(detid) ? modulesInRingFront_[tTopo_->tecRing(detid)] : 0)));

  xyb.x = lowX_ + xyb.ix - 0.5;
  xyb.y = lowY_ + xyb.iy - 0.5;

  return xyb;
}

// TkDetMap

std::string TkDetMap::getLayerName(int in) {
  switch (in) {
    case TkLayerMap::TIB_L1:
      return "TIB_L1";
    case TkLayerMap::TIB_L2:
      return "TIB_L2";
    case TkLayerMap::TIB_L3:
      return "TIB_L3";
    case TkLayerMap::TIB_L4:
      return "TIB_L4";
    case TkLayerMap::TIDP_D1:
      return "TIDP_D1";
    case TkLayerMap::TIDP_D2:
      return "TIDP_D2";
    case TkLayerMap::TIDP_D3:
      return "TIDP_D3";
    case TkLayerMap::TIDM_D1:
      return "TIDM_D1";
    case TkLayerMap::TIDM_D2:
      return "TIDM_D2";
    case TkLayerMap::TIDM_D3:
      return "TIDM_D3";
    case TkLayerMap::TOB_L1:
      return "TOB_L1";
    case TkLayerMap::TOB_L2:
      return "TOB_L2";
    case TkLayerMap::TOB_L3:
      return "TOB_L3";
    case TkLayerMap::TOB_L4:
      return "TOB_L4";
    case TkLayerMap::TOB_L5:
      return "TOB_L5";
    case TkLayerMap::TOB_L6:
      return "TOB_L6";
    case TkLayerMap::TECP_W1:
      return "TECP_W1";
    case TkLayerMap::TECP_W2:
      return "TECP_W2";
    case TkLayerMap::TECP_W3:
      return "TECP_W3";
    case TkLayerMap::TECP_W4:
      return "TECP_W4";
    case TkLayerMap::TECP_W5:
      return "TECP_W5";
    case TkLayerMap::TECP_W6:
      return "TECP_W6";
    case TkLayerMap::TECP_W7:
      return "TECP_W7";
    case TkLayerMap::TECP_W8:
      return "TECP_W8";
    case TkLayerMap::TECP_W9:
      return "TECP_W9";
    case TkLayerMap::TECM_W1:
      return "TECM_W1";
    case TkLayerMap::TECM_W2:
      return "TECM_W2";
    case TkLayerMap::TECM_W3:
      return "TECM_W3";
    case TkLayerMap::TECM_W4:
      return "TECM_W4";
    case TkLayerMap::TECM_W5:
      return "TECM_W5";
    case TkLayerMap::TECM_W6:
      return "TECM_W6";
    case TkLayerMap::TECM_W7:
      return "TECM_W7";
    case TkLayerMap::TECM_W8:
      return "TECM_W8";
    case TkLayerMap::TECM_W9:
      return "TECM_W9";
  }
  return "Invalid";
}

int TkDetMap::getLayerNum(const std::string& in) {
  if (in == "TIB_L1")
    return TkLayerMap::TIB_L1;
  if (in == "TIB_L2")
    return TkLayerMap::TIB_L2;
  if (in == "TIB_L3")
    return TkLayerMap::TIB_L3;
  if (in == "TIB_L4")
    return TkLayerMap::TIB_L4;
  if (in == "TIDP_D1")
    return TkLayerMap::TIDP_D1;
  if (in == "TIDP_D2")
    return TkLayerMap::TIDP_D2;
  if (in == "TIDP_D3")
    return TkLayerMap::TIDP_D3;
  if (in == "TIDM_D1")
    return TkLayerMap::TIDM_D1;
  if (in == "TIDM_D2")
    return TkLayerMap::TIDM_D2;
  if (in == "TIDM_D3")
    return TkLayerMap::TIDM_D3;
  if (in == "TOB_L1")
    return TkLayerMap::TOB_L1;
  if (in == "TOB_L2")
    return TkLayerMap::TOB_L2;
  if (in == "TOB_L3")
    return TkLayerMap::TOB_L3;
  if (in == "TOB_L4")
    return TkLayerMap::TOB_L4;
  if (in == "TOB_L5")
    return TkLayerMap::TOB_L5;
  if (in == "TOB_L6")
    return TkLayerMap::TOB_L6;
  if (in == "TECP_W1")
    return TkLayerMap::TECP_W1;
  if (in == "TECP_W2")
    return TkLayerMap::TECP_W2;
  if (in == "TECP_W3")
    return TkLayerMap::TECP_W3;
  if (in == "TECP_W4")
    return TkLayerMap::TECP_W4;
  if (in == "TECP_W5")
    return TkLayerMap::TECP_W5;
  if (in == "TECP_W6")
    return TkLayerMap::TECP_W6;
  if (in == "TECP_W7")
    return TkLayerMap::TECP_W7;
  if (in == "TECP_W8")
    return TkLayerMap::TECP_W8;
  if (in == "TECP_W9")
    return TkLayerMap::TECP_W9;
  if (in == "TECM_W1")
    return TkLayerMap::TECM_W1;
  if (in == "TECM_W2")
    return TkLayerMap::TECM_W2;
  if (in == "TECM_W3")
    return TkLayerMap::TECM_W3;
  if (in == "TECM_W4")
    return TkLayerMap::TECM_W4;
  if (in == "TECM_W5")
    return TkLayerMap::TECM_W5;
  if (in == "TECM_W6")
    return TkLayerMap::TECM_W6;
  if (in == "TECM_W7")
    return TkLayerMap::TECM_W7;
  if (in == "TECM_W8")
    return TkLayerMap::TECM_W8;
  if (in == "TECM_W9")
    return TkLayerMap::TECM_W9;
  return 0;
}

void TkDetMap::getSubDetLayerSide(int in, SiStripDetId::SubDetector& subDet, uint32_t& layer, uint32_t& side) {
  switch (in) {
    case TkLayerMap::TIB_L1:
      subDet = SiStripDetId::TIB;
      layer = 1;
      break;
    case TkLayerMap::TIB_L2:
      subDet = SiStripDetId::TIB;
      layer = 2;
      break;
    case TkLayerMap::TIB_L3:
      subDet = SiStripDetId::TIB;
      layer = 3;
      break;
    case TkLayerMap::TIB_L4:
      subDet = SiStripDetId::TIB;
      layer = 4;
      break;
    case TkLayerMap::TIDP_D1:
      subDet = SiStripDetId::TID;
      layer = 1;
      side = 2;
      break;
    case TkLayerMap::TIDP_D2:
      subDet = SiStripDetId::TID;
      layer = 2;
      side = 2;
      break;
    case TkLayerMap::TIDP_D3:
      subDet = SiStripDetId::TID;
      layer = 3;
      side = 2;
      break;
    case TkLayerMap::TIDM_D1:
      subDet = SiStripDetId::TID;
      layer = 1;
      side = 1;
      break;
    case TkLayerMap::TIDM_D2:
      subDet = SiStripDetId::TID;
      layer = 2;
      side = 1;
      break;
    case TkLayerMap::TIDM_D3:
      subDet = SiStripDetId::TID;
      layer = 3;
      side = 1;
      break;
    case TkLayerMap::TOB_L1:
      subDet = SiStripDetId::TOB;
      layer = 1;
      break;
    case TkLayerMap::TOB_L2:
      subDet = SiStripDetId::TOB;
      layer = 2;
      break;
    case TkLayerMap::TOB_L3:
      subDet = SiStripDetId::TOB;
      layer = 3;
      break;
    case TkLayerMap::TOB_L4:
      subDet = SiStripDetId::TOB;
      layer = 4;
      break;
    case TkLayerMap::TOB_L5:
      subDet = SiStripDetId::TOB;
      layer = 5;
      break;
    case TkLayerMap::TOB_L6:
      subDet = SiStripDetId::TOB;
      layer = 6;
      break;
    case TkLayerMap::TECP_W1:
      subDet = SiStripDetId::TEC;
      layer = 1;
      side = 2;
      break;
    case TkLayerMap::TECP_W2:
      subDet = SiStripDetId::TEC;
      layer = 2;
      side = 2;
      break;
    case TkLayerMap::TECP_W3:
      subDet = SiStripDetId::TEC;
      layer = 3;
      side = 2;
      break;
    case TkLayerMap::TECP_W4:
      subDet = SiStripDetId::TEC;
      layer = 4;
      side = 2;
      break;
    case TkLayerMap::TECP_W5:
      subDet = SiStripDetId::TEC;
      layer = 5;
      side = 2;
      break;
    case TkLayerMap::TECP_W6:
      subDet = SiStripDetId::TEC;
      layer = 6;
      side = 2;
      break;
    case TkLayerMap::TECP_W7:
      subDet = SiStripDetId::TEC;
      layer = 7;
      side = 2;
      break;
    case TkLayerMap::TECP_W8:
      subDet = SiStripDetId::TEC;
      layer = 8;
      side = 2;
      break;
    case TkLayerMap::TECP_W9:
      subDet = SiStripDetId::TEC;
      layer = 9;
      side = 2;
      break;
    case TkLayerMap::TECM_W1:
      subDet = SiStripDetId::TEC;
      layer = 1;
      side = 1;
      break;
    case TkLayerMap::TECM_W2:
      subDet = SiStripDetId::TEC;
      layer = 2;
      side = 1;
      break;
    case TkLayerMap::TECM_W3:
      subDet = SiStripDetId::TEC;
      layer = 3;
      side = 1;
      break;
    case TkLayerMap::TECM_W4:
      subDet = SiStripDetId::TEC;
      layer = 4;
      side = 1;
      break;
    case TkLayerMap::TECM_W5:
      subDet = SiStripDetId::TEC;
      layer = 5;
      side = 1;
      break;
    case TkLayerMap::TECM_W6:
      subDet = SiStripDetId::TEC;
      layer = 6;
      side = 1;
      break;
    case TkLayerMap::TECM_W7:
      subDet = SiStripDetId::TEC;
      layer = 7;
      side = 1;
      break;
    case TkLayerMap::TECM_W8:
      subDet = SiStripDetId::TEC;
      layer = 8;
      side = 1;
      break;
    case TkLayerMap::TECM_W9:
      subDet = SiStripDetId::TEC;
      layer = 9;
      side = 1;
      break;
  }
}

const TkLayerMap::XYbin& TkDetMap::getXY(DetId detid,
                                         DetId& cached_detid,
                                         int16_t& cached_layer,
                                         TkLayerMap::XYbin& cached_XYbin) const {
  LogTrace("TkDetMap") << "[getXY] detid " << detid.rawId() << " cache " << cached_detid.rawId() << " layer "
                       << cached_layer << " XY " << cached_XYbin.ix << " " << cached_XYbin.iy << " " << cached_XYbin.x
                       << " " << cached_XYbin.y;
  if (detid == cached_detid)
    return cached_XYbin;

  /*FIXME*/
  //if (layer!=INVALID)
  findLayer(detid, cached_detid, cached_layer, cached_XYbin);
  LogTrace("TkDetMap") << "[getXY] detid " << detid.rawId() << " cache " << cached_detid.rawId() << " layer "
                       << cached_layer << " XY " << cached_XYbin.ix << " " << cached_XYbin.iy << " " << cached_XYbin.x
                       << " " << cached_XYbin.y;

  return cached_XYbin;
}

int16_t TkDetMap::findLayer(DetId detid,
                            DetId& cached_detid,
                            int16_t& cached_layer,
                            TkLayerMap::XYbin& cached_XYbin) const {
  if (detid == cached_detid)
    return cached_layer;

  cached_detid = detid;

  int16_t layer = TkLayerMap::layerSearch(detid, tTopo_);
  if (layer == TkLayerMap::INVALID) {
    // there is something wrong if the layer is 0, early return
    return TkLayerMap::INVALID;
  }

  LogTrace("TkDetMap") << "[findLayer] detid " << detid.rawId() << " layer " << layer;
  if (layer != cached_layer) {
    cached_layer = layer;
  }
  cached_XYbin = TkMap[cached_layer].getXY(detid, layer);
  LogTrace("TkDetMap") << "[findLayer] detid " << detid.rawId() << " cached_XYbin " << cached_XYbin.ix << " "
                       << cached_XYbin.iy;

  return cached_layer;
}

void TkDetMap::getComponents(
    int layer, int& nchX, double& lowX, double& highX, int& nchY, double& lowY, double& highY) const {
  nchX = TkMap[layer].get_nchX();
  lowX = TkMap[layer].get_lowX();
  highX = TkMap[layer].get_highX();
  nchY = TkMap[layer].get_nchY();
  lowY = TkMap[layer].get_lowY();
  highY = TkMap[layer].get_highY();
}
