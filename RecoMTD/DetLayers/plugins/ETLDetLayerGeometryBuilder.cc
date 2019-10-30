#include "ETLDetLayerGeometryBuilder.h"

#include <RecoMTD/DetLayers/interface/MTDRingForwardDoubleLayer.h>
#include <RecoMTD/DetLayers/interface/MTDDetRing.h>
#include <DataFormats/ForwardDetId/interface/ETLDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

pair<vector<DetLayer*>, vector<DetLayer*> > ETLDetLayerGeometryBuilder::buildLayers(const MTDGeometry& geo) {
  vector<DetLayer*> result[2];  // one for each endcap

  for (unsigned endcap = 0; endcap < 2; ++endcap) {
    // there is only one layer for ETL right now, maybe more later
    for (unsigned layer = 0; layer <= 0; ++layer) {
      vector<unsigned> rings;
      for (unsigned ring = 1; ring <= 12; ++ring) {
        rings.push_back(ring);
      }
      MTDRingForwardDoubleLayer* thelayer = buildLayer(endcap, layer, rings, geo);
      if (thelayer)
        result[endcap].push_back(thelayer);
    }
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]);
  return res_pair;
}

MTDRingForwardDoubleLayer* ETLDetLayerGeometryBuilder::buildLayer(int endcap,
                                                                  int layer,
                                                                  vector<unsigned>& rings,
                                                                  const MTDGeometry& geo) {
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|ETLDetLayerGeometryBuilder";
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
        LogTrace(metname) << "get ETL module " << std::hex << ETLDetId(endcap, layer, ring, module).rawId() << std::dec
                          << " at R=" << geomDet->position().perp() << ", phi=" << geomDet->position().phi()
                          << ", z= " << geomDet->position().z() << " isFront? " << isInFront << std::endl;
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
  LogTrace(metname) << "New MTDRingForwardLayer with " << frontRings.size() << " and " << backRings.size()
                    << " rings, at Z " << result->position().z() << " R1: " << result->specificSurface().innerRadius()
                    << " R2: " << result->specificSurface().outerRadius() << std::endl;
  return result;
}

bool ETLDetLayerGeometryBuilder::isFront(int layer, int ring, int module) { return (module + 1) % 2; }

MTDDetRing* ETLDetLayerGeometryBuilder::makeDetRing(vector<const GeomDet*>& geomDets) {
  const std::string metname = "MTD|RecoMTD|RecoMTDDetLayers|ETLDetLayerGeometryBuilder";

  precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
  MTDDetRing* result = new MTDDetRing(geomDets);
  LogTrace(metname) << "New MTDDetRing with " << geomDets.size() << " chambers at z=" << result->position().z()
                    << " R1: " << result->specificSurface().innerRadius()
                    << " R2: " << result->specificSurface().outerRadius() << std::endl;
  ;
  return result;
}
