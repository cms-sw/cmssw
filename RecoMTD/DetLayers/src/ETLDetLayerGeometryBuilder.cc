//#define EDM_ML_DEBUG

#include "RecoMTD/DetLayers/interface/ETLDetLayerGeometryBuilder.h"

#include <RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetSector.h>
#include <DataFormats/ForwardDetId/interface/ETLDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/MTDCommonData/interface/MTDTopologyMode.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

pair<vector<DetLayer*>, vector<DetLayer*> > ETLDetLayerGeometryBuilder::buildLayers(const MTDGeometry& geo,
                                                                                    const MTDTopology& topo) {
  vector<DetLayer*> result[2];  // one for each endcap

  const int mtdTopologyMode = topo.getMTDTopologyMode();
  ETLDetId::EtlLayout etlL = MTDTopologyMode::etlLayoutFromTopoMode(mtdTopologyMode);
  // number of layers is identical for post TDR scenarios, pick v4
  // loop on number of sectors per face, two faces per disc (i.e. layer) taken into account in layer building (front/back)
  unsigned int nSector(1);
  if (etlL == ETLDetId::EtlLayout::v4) {
    nSector *= ETLDetId::kETLv4maxSector;
  } else if (etlL == ETLDetId::EtlLayout::v5) {
    nSector *= ETLDetId::kETLv5maxSector;
  } else if (etlL == ETLDetId::EtlLayout::v8) {
    nSector *= ETLDetId::kETLv5maxSector;
  } else {
    throw cms::Exception("MTDDetLayers") << "Not implemented scenario " << mtdTopologyMode;
  }

  for (unsigned endcap = 0; endcap < 2; ++endcap) {
    // number of layers is two, identical for post TDR scenarios, pick v4
    for (unsigned layer = 1; layer <= ETLDetId::kETLv4nDisc; ++layer) {
      vector<unsigned> sectors;
      sectors.reserve(nSector + 1);
      for (unsigned sector = 1; sector <= nSector; ++sector) {
        sectors.push_back(sector);
      }
      MTDSectorForwardDoubleLayer* thelayer = buildLayer(endcap, layer, sectors, geo, topo);
      if (thelayer)
        result[endcap].push_back(thelayer);
    }
  }
  //
  // the first entry is Z+ ( MTD side 1), the second is Z- (MTD side 0)
  //
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[1], result[0]);
  return res_pair;
}

bool ETLDetLayerGeometryBuilder::isFront(int layer, int ring, int module) { return (module + 1) % 2; }

MTDSectorForwardDoubleLayer* ETLDetLayerGeometryBuilder::buildLayer(
    int endcap, int layer, vector<unsigned>& sectors, const MTDGeometry& geo, const MTDTopology& topo) {
  MTDSectorForwardDoubleLayer* result = nullptr;

  std::vector<const MTDDetSector*> frontSectors, backSectors;

  LogDebug("MTDDetLayers") << "ETL dets array size = " << geo.detsETL().size();

  for (unsigned sector : sectors) {
    std::vector<const GeomDet*> frontGeomDets, backGeomDets;
    LogDebug("MTDDetLayers") << "endcap = " << endcap << " layer = " << layer << " sector = " << sector;
#ifdef EDM_ML_DEBUG
    unsigned int nfront(0), nback(0);
#endif
    for (auto det : geo.detsETL()) {
      ETLDetId theMod(det->geographicalId().rawId());
      if (theMod.mtdSide() == endcap && theMod.nDisc() == layer && theMod.sector() == static_cast<int>(sector)) {
        LogTrace("MTDLayerDump") << std::fixed << "ETLDetId " << theMod.rawId() << " side = " << std::setw(4)
                                 << theMod.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << theMod.nDisc() << " "
                                 << std::setw(4) << theMod.discSide() << " " << std::setw(4) << theMod.sector()
                                 << " mod/type = " << std::setw(4) << theMod.module() << " " << std::setw(4)
                                 << theMod.modType() << " pos = " << det->position();
        // front layer face
        if (theMod.discSide() == 0) {
#ifdef EDM_ML_DEBUG
          nfront++;
          LogTrace("MTDDetLayers") << "Front " << theMod.discSide() << " " << nfront;
#endif
          frontGeomDets.emplace_back(det);
          // back layer face
        } else if (theMod.discSide() == 1) {
#ifdef EDM_ML_DEBUG
          nback++;
          LogTrace("MTDDetLayers") << "Back " << theMod.discSide() << " " << nback;
#endif
          backGeomDets.emplace_back(det);
        }
      }
    }

    if (!backGeomDets.empty()) {
      std::sort(backGeomDets.begin(), backGeomDets.end(), topo.orderETLSector);
      LogDebug("MTDDetLayers") << "backGeomDets size = " << backGeomDets.size();
      backSectors.emplace_back(makeDetSector(backGeomDets, topo));
    }

    if (!frontGeomDets.empty()) {
      std::sort(frontGeomDets.begin(), frontGeomDets.end(), topo.orderETLSector);
      LogDebug("MTDDetLayers") << "frontGeomDets size = " << frontGeomDets.size();
      frontSectors.emplace_back(makeDetSector(frontGeomDets, topo));
      assert(!backGeomDets.empty());
      float frontz = frontSectors.back()->position().z();
      float backz = backSectors.back()->position().z();
      assert(std::abs(frontz) < std::abs(backz));
    }
  }

  result = new MTDSectorForwardDoubleLayer(frontSectors, backSectors);
  LogTrace("MTDDetLayers") << "New MTDSectorForwardDoubleLayer with " << std::fixed << std::setw(14)
                           << frontSectors.size() << " and " << std::setw(14) << backSectors.size() << " rings, at Z "
                           << std::setw(14) << result->specificSurface().position().z() << " R1: " << std::setw(14)
                           << result->specificSurface().innerRadius() << " R2: " << std::setw(14)
                           << result->specificSurface().outerRadius();

  return result;
}

MTDDetSector* ETLDetLayerGeometryBuilder::makeDetSector(vector<const GeomDet*>& geomDets, const MTDTopology& topo) {
  MTDDetSector* result = new MTDDetSector(geomDets, topo);
  LogTrace("MTDDetLayers") << "ETLDetLayerGeometryBuilder::makeDetSector new MTDDetSector with " << std::fixed
                           << std::setw(14) << geomDets.size() << " modules \n"
                           << (*result);

  return result;
}
