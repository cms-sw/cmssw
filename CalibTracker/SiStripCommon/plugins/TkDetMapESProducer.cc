#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
class TrackerTopology;

#include "Geometry/TrackerNumberingBuilder/interface/utils.h"

class TkDetMapESProducer : public edm::ESProducer {
public:
  TkDetMapESProducer(const edm::ParameterSet&);
  ~TkDetMapESProducer() override {}

  std::unique_ptr<TkDetMap> produce(const TrackerTopologyRcd&);

private:
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
};

TkDetMapESProducer::TkDetMapESProducer(const edm::ParameterSet&) {
  auto cc = setWhatProduced(this);
  tTopoToken_ = cc.consumes();
  geomDetToken_ = cc.consumes();
}

namespace {
  TkLayerMap makeTkLayerMap(int layer, const TrackerTopology* tTopo, const std::vector<uint32_t> tkDetIdList) {
    LogTrace("TkLayerMap") << " TkLayerMap::constructor for layer " << layer;
    uint32_t nStringExt, nRod;
    std::vector<uint32_t> SingleExtString;
    switch (layer) {
      case TkLayerMap::TIB_L1:  //TIBL1
        nStringExt = 30;
        SingleExtString.insert(SingleExtString.begin(), 8, 0);
        SingleExtString.insert(SingleExtString.begin() + 8, 7, 1);
        SingleExtString.insert(SingleExtString.begin() + 15, 8, 2);
        SingleExtString.insert(SingleExtString.begin() + 23, 7, 3);
        return TkLayerMap(layer,
                          12,
                          -6.,
                          6.,
                          2 * (nStringExt + 1),
                          -1. * (nStringExt + 1.),
                          (nStringExt + 1),
                          tTopo,
                          tkDetIdList,
                          SingleExtString,
                          {},
                          {},
                          {},
                          nStringExt);
        break;
      case TkLayerMap::TIB_L2:
        nStringExt = 38;
        SingleExtString.insert(SingleExtString.begin(), 10, 0);
        SingleExtString.insert(SingleExtString.begin() + 10, 9, 1);
        SingleExtString.insert(SingleExtString.begin() + 19, 10, 2);
        SingleExtString.insert(SingleExtString.begin() + 29, 9, 3);
        return TkLayerMap(layer,
                          12,
                          -6.,
                          6.,
                          2 * (nStringExt + 1),
                          -1. * (nStringExt + 1.),
                          (nStringExt + 1),
                          tTopo,
                          tkDetIdList,
                          SingleExtString,
                          {},
                          {},
                          {},
                          nStringExt);
        break;
      case TkLayerMap::TIB_L3:
        nStringExt = 46;
        SingleExtString.insert(SingleExtString.begin(), 23, 0);
        SingleExtString.insert(SingleExtString.begin() + 23, 23, 1);
        return TkLayerMap(
            layer, 12, -6., 6., nStringExt, 0, nStringExt, tTopo, tkDetIdList, SingleExtString, {}, {}, {}, nStringExt);
        break;
      case TkLayerMap::TIB_L4:
        nStringExt = 56;
        SingleExtString.insert(SingleExtString.begin(), 14, 0);
        SingleExtString.insert(SingleExtString.begin() + 14, 14, 1);
        SingleExtString.insert(SingleExtString.begin() + 28, 14, 2);
        SingleExtString.insert(SingleExtString.begin() + 42, 14, 3);
        return TkLayerMap(
            layer, 12, -6., 6., nStringExt, 0, nStringExt, tTopo, tkDetIdList, SingleExtString, {}, {}, {}, nStringExt);
        break;
      case TkLayerMap::TIDM_D1:  //TID
      case TkLayerMap::TIDM_D2:  //TID
      case TkLayerMap::TIDM_D3:  //TID
        return TkLayerMap(layer, 7, -7., 0., 40, 0., 40., tTopo, tkDetIdList, {});
        break;
      case TkLayerMap::TIDP_D1:  //TID
      case TkLayerMap::TIDP_D2:  //TID
      case TkLayerMap::TIDP_D3:  //TID
        return TkLayerMap(layer, 7, 0., 7., 40, 0., 40., tTopo, tkDetIdList, {});
        break;
      case TkLayerMap::TOB_L1:  //TOBL1
        nRod = 42;
        return TkLayerMap(layer,
                          12,
                          -6.,
                          6.,
                          2 * (nRod + 1),
                          -1. * (nRod + 1.),
                          (nRod + 1.),
                          tTopo,
                          tkDetIdList,
                          {},
                          {},
                          {},
                          {},
                          0,
                          nRod);
        break;
      case TkLayerMap::TOB_L2:
        nRod = 48;
        return TkLayerMap(layer,
                          12,
                          -6.,
                          6.,
                          2 * (nRod + 1),
                          -1. * (nRod + 1.),
                          (nRod + 1.),
                          tTopo,
                          tkDetIdList,
                          {},
                          {},
                          {},
                          {},
                          0,
                          nRod);
        break;
      case TkLayerMap::TOB_L3:  //TOBL3
        nRod = 54;
        return TkLayerMap(layer, 12, -6., 6., nRod, 0., 1. * nRod, tTopo, tkDetIdList, {}, {}, {}, {}, 0, nRod);
        break;
      case TkLayerMap::TOB_L4:
        nRod = 60;
        return TkLayerMap(layer, 12, -6., 6., nRod, 0., 1. * nRod, tTopo, tkDetIdList, {}, {}, {}, {}, 0, nRod);
        break;
      case TkLayerMap::TOB_L5:
        nRod = 66;
        return TkLayerMap(layer, 12, -6., 6., nRod, 0., 1. * nRod, tTopo, tkDetIdList, {}, {}, {}, {}, 0, nRod);
        break;
      case TkLayerMap::TOB_L6:
        nRod = 74;
        return TkLayerMap(layer, 12, -6., 6., nRod, 0., 1. * nRod, tTopo, tkDetIdList, {}, {}, {}, {}, 0, nRod);
        break;
      default:  //TEC
        std::vector<uint32_t> modulesInRingFront = {0, 2, 2, 3, 4, 2, 4, 5};
        std::vector<uint32_t> modulesInRingBack = {0, 1, 1, 2, 3, 3, 3, 5};
        switch (layer) {
          case TkLayerMap::TECM_W1:
          case TkLayerMap::TECM_W2:
          case TkLayerMap::TECM_W3:
            return TkLayerMap(layer,
                              16,
                              -16.,
                              0.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 1, 4, 7, 9, 11, 14, 16});
            break;
          case TkLayerMap::TECM_W4:
          case TkLayerMap::TECM_W5:
          case TkLayerMap::TECM_W6:
            return TkLayerMap(layer,
                              13,
                              -16.,
                              -3.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 0, 1, 4, 6, 8, 11, 13});
            break;
          case TkLayerMap::TECM_W7:
          case TkLayerMap::TECM_W8:
            return TkLayerMap(layer,
                              10,
                              -16.,
                              -6.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 0, 0, 1, 3, 5, 8, 10});
            break;
          case TkLayerMap::TECM_W9:
            return TkLayerMap(layer,
                              8,
                              -16.,
                              -8.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 0, 0, 0, 1, 3, 6, 8});
            break;
          case TkLayerMap::TECP_W1:
          case TkLayerMap::TECP_W2:
          case TkLayerMap::TECP_W3:
            return TkLayerMap(layer,
                              16,
                              0.,
                              16.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 1, 4, 7, 9, 11, 14, 16});
            break;
          case TkLayerMap::TECP_W4:
          case TkLayerMap::TECP_W5:
          case TkLayerMap::TECP_W6:
            return TkLayerMap(layer,
                              13,
                              3.,
                              16.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 0, 1, 4, 6, 8, 11, 13});
            break;
          case TkLayerMap::TECP_W7:
          case TkLayerMap::TECP_W8:
            return TkLayerMap(layer,
                              10,
                              6.,
                              16.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 0, 0, 1, 3, 5, 8, 10});
            break;
          case TkLayerMap::TECP_W9:
            return TkLayerMap(layer,
                              8,
                              8.,
                              16.,
                              80,
                              0.,
                              80.,
                              tTopo,
                              tkDetIdList,
                              {},
                              modulesInRingFront,
                              modulesInRingBack,
                              {0, 0, 0, 0, 1, 3, 6, 8});
        }
    }
    return TkLayerMap{};
  }
}  // namespace

std::unique_ptr<TkDetMap> TkDetMapESProducer::produce(const TrackerTopologyRcd& tTopoRcd) {
  const auto& geomDet = tTopoRcd.getRecord<IdealGeometryRecord>().get(geomDetToken_);
  const auto TkDetIdList = TrackerGeometryUtils::getSiStripDetIds(geomDet);

  const auto& tTopo = tTopoRcd.get(tTopoToken_);
  auto tkDetMap = std::make_unique<TkDetMap>(&tTopo);

  LogTrace("TkDetMap") << "TkDetMap::constructor ";
  //Create TkLayerMap for each layer declared in the TkLayerEnum
  for (int layer = 1; layer < TkLayerMap::NUMLAYERS; ++layer) {
    tkDetMap->setLayerMap(layer, makeTkLayerMap(layer, &tTopo, TkDetIdList));
  }

  return tkDetMap;
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(TkDetMapESProducer);
