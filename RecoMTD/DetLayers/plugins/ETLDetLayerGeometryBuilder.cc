#define EDM_ML_DEBUG

#include "ETLDetLayerGeometryBuilder.h"

#include <RecoMTD/DetLayers/interface/MTDRingForwardDoubleLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetRing.h>
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
                                                                                    const int mtdTopologyMode) {
  vector<DetLayer*> result[2];  // one for each endcap

  if (mtdTopologyMode <= static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    for (unsigned endcap = 0; endcap < 2; ++endcap) {
      // there is only one layer for ETL right now, maybe more later
      for (unsigned layer = 0; layer < ETLDetId::kETLv1nDisc; ++layer) {
        vector<unsigned> rings;
        for (unsigned ring = 1; ring <= ETLDetId::kETLv1maxRing; ++ring) {
          rings.push_back(ring);
        }
        MTDRingForwardDoubleLayer* thelayer = buildLayer(endcap, layer, rings, geo);
        if (thelayer)
          result[endcap].push_back(thelayer);
      }
    }
  } else {
    // number of layers is identical for post TDR scenarios, pick v4
    // loop on number of sectors per face, two faces per disc (i.e. layer) taken into account in layer building (front/back)
    unsigned int nSector(1);
    switch (mtdTopologyMode) {
      case static_cast<int>(MTDTopologyMode::Mode::btlv1etlv4):
        nSector *= ETLDetId::kETLv4maxSector;
        break;
      //case static_cast<int>(MTDTopologyMode::Mode::btlv1etlv5):
      //nSector *= ETLDetId::kETLv5maxSector;
      //break;
      default:
        throw cms::Exception("MTDDetLayers") << "Not implemented scenario " << mtdTopologyMode;
        break;
    }

    for (unsigned endcap = 0; endcap < 2; ++endcap) {
      // number of layers is two, identical for post TDR scenarios, pick v4
      for (unsigned layer = 1; layer <= ETLDetId::kETLv4nDisc; ++layer) {
        vector<unsigned> sectors;
        for (unsigned sector = 1; sector <= nSector; ++sector) {
          sectors.push_back(sector);
        }
        MTDSectorForwardDoubleLayer* thelayer = buildLayerNew(endcap, layer, sectors, geo);
        if (thelayer)
          result[endcap].push_back(thelayer);
      }
    }
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]);
  return res_pair;
}

MTDRingForwardDoubleLayer* ETLDetLayerGeometryBuilder::buildLayer(int endcap,
                                                                  int layer,
                                                                  vector<unsigned>& rings,
                                                                  const MTDGeometry& geo) {
  MTDRingForwardDoubleLayer* result = nullptr;

  vector<const ForwardDetRing*> frontRings, backRings;

  for (unsigned ring : rings) {
    vector<const GeomDet*> frontGeomDets, backGeomDets;
    for (unsigned module = 1; module <= ETLDetId::kETLmoduleMask; ++module) {
      ETLDetId detId(endcap, ring, module, 0);
      const GeomDet* geomDet = geo.idToDet(detId);
      // we sometimes loop over more chambers than there are in ring
      bool isInFront = isFront(layer, ring, module);
      if (geomDet != nullptr) {
        if (isInFront) {
          frontGeomDets.push_back(geomDet);
        } else {
          backGeomDets.push_back(geomDet);
        }
        LogTrace("MTDDetLayers") << "get ETL module " << std::hex << ETLDetId(endcap, layer, ring, module).rawId()
                                 << std::dec << " at R=" << geomDet->position().perp()
                                 << ", phi=" << geomDet->position().phi() << ", z= " << geomDet->position().z()
                                 << " isFront? " << isInFront;
      }
    }

    if (!backGeomDets.empty()) {
      backRings.push_back(makeDetRing(backGeomDets));
    }

    if (!frontGeomDets.empty()) {
      frontRings.push_back(makeDetRing(frontGeomDets));
      assert(!backGeomDets.empty());
      float frontz = frontRings[0]->position().z();
      float backz = backRings[0]->position().z();
      assert(fabs(frontz) < fabs(backz));
    }
  }

  // How should they be sorted?
  //    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractZ<GeometricSearchDet,float>());
  result = new MTDRingForwardDoubleLayer(frontRings, backRings);
  LogTrace("MTDDetLayers") << "New MTDRingForwardLayer with " << frontRings.size() << " and " << backRings.size()
                           << " rings, at Z " << result->position().z()
                           << " R1: " << result->specificSurface().innerRadius()
                           << " R2: " << result->specificSurface().outerRadius();
  return result;
}

bool ETLDetLayerGeometryBuilder::isFront(int layer, int ring, int module) { return (module + 1) % 2; }

MTDDetRing* ETLDetLayerGeometryBuilder::makeDetRing(vector<const GeomDet*>& geomDets) {
  precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
  MTDDetRing* result = new MTDDetRing(geomDets);
  LogTrace("MTDDetLayers") << "ETLDetLayerGeometryBuilder: new MTDDetRing with " << geomDets.size()
                           << " chambers at z=" << result->position().z()
                           << " R1: " << result->specificSurface().innerRadius()
                           << " R2: " << result->specificSurface().outerRadius();
  return result;
}

MTDSectorForwardDoubleLayer* ETLDetLayerGeometryBuilder::buildLayerNew(int endcap,
                                                                       int layer,
                                                                       vector<unsigned>& sectors,
                                                                       const MTDGeometry& geo) {
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
        LogDebug("MTDDetLayers") << "ETLDetId " << theMod.rawId() << " side = " << theMod.mtdSide()
                                 << " Disc/Side/Sector = " << theMod.nDisc() << " " << theMod.discSide() << " "
                                 << theMod.sector() << " mod/type = " << theMod.module() << " " << theMod.modType()
                                 << " pos = " << det->position();
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
      LogDebug("MTDDetLayers") << "backGeomDets size = " << backGeomDets.size();
      backSectors.emplace_back(makeDetSector(backGeomDets));
    }

    if (!frontGeomDets.empty()) {
      LogDebug("MTDDetLayers") << "frontGeomDets size = " << frontGeomDets.size();
      frontSectors.emplace_back(makeDetSector(frontGeomDets));
      assert(!backGeomDets.empty());
      float frontz = frontSectors.back()->position().z();
      float backz = backSectors.back()->position().z();
      assert(fabs(frontz) < fabs(backz));
    }
  }

  result = new MTDSectorForwardDoubleLayer(frontSectors, backSectors);
  LogTrace("MTDDetLayers") << "New MTDSectorForwardDoubleLayer with " << frontSectors.size() << " and "
                           << backSectors.size() << " rings, at Z " << result->position().z()
                           << " R1: " << result->specificSurface().innerRadius()
                           << " R2: " << result->specificSurface().outerRadius();

  return result;
}

MTDDetSector* ETLDetLayerGeometryBuilder::makeDetSector(vector<const GeomDet*>& geomDets) {
  LogTrace("MTDDetLayers") << "ETLDetLayerGeometryBuilder: new MTDDetSector with " << geomDets.size() << " modules";

  MTDDetSector* result = new MTDDetSector(geomDets);
  LogTrace("MTDDetLayers") << "ETLDetLayerGeometryBuilder: pos = " << result->position()
                           << " rmin = " << result->specificSurface().innerRadius()
                           << " rmax = " << result->specificSurface().outerRadius()
                           << " phi ref = " << result->specificSurface().position().phi()
                           << " phi/2 = " << result->specificSurface().phiHalfExtension();

  return result;
}
