#include <RecoMuon/DetLayers/src/MuonGEMDetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <TrackingTools/DetLayers/interface/ForwardDetLayer.h>
#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h>
#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <RecoMuon/DetLayers/interface/MuDetRod.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonGEMDetLayerGeometryBuilder::~MuonGEMDetLayerGeometryBuilder() {}

// Builds the forward (first) and backward (second) layers
// Builds etaPartitions (for rechits)
pair<vector<DetLayer*>, vector<DetLayer*> > MuonGEMDetLayerGeometryBuilder::buildEndcapLayers(const GEMGeometry& geo) {
  vector<DetLayer*> result[2];
  for (int endcap = -1; endcap <= 1; endcap += 2) {
    int iendcap = (endcap == 1) ? 0 : 1;  // +1: forward, -1: backward

    for (int station = GEMDetId::minStationId0; station <= GEMDetId::maxStationId; ++station) {
      for (int layer = GEMDetId::minLayerId + 1; layer <= GEMDetId::maxLayerId0; ++layer) {
        if (station != GEMDetId::minStationId0 && layer > GEMDetId::maxLayerId)
          break;
        vector<int> rolls, rings, chambers;
        rings.push_back(GEMDetId::minRingId);
        for (int chamber = GEMDetId::minChamberId + 1; chamber <= GEMDetId::maxChamberId; chamber++) {
          chambers.push_back(chamber);
          // ME0 is composed of 18 super-chambers
          if (station == GEMDetId::minStationId0 && chamber == GEMDetId::maxChamberId / 2)
            break;
        }
        for (int roll = GEMDetId::minRollId + 1; roll <= GEMDetId::maxRollId; ++roll) {
          rolls.push_back(roll);
          // ME0 layer consists of 10 etapartitions
          if (station == GEMDetId::minStationId0 && roll == 10)
            break;
        }
        ForwardDetLayer* ringLayer = buildLayer(endcap, rings, station, layer, chambers, rolls, geo);
        if (ringLayer)
          result[iendcap].push_back(ringLayer);
      }
    }
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]);
  return res_pair;
}

ForwardDetLayer* MuonGEMDetLayerGeometryBuilder::buildLayer(int endcap,
                                                            vector<int>& rings,
                                                            int station,
                                                            int layer,
                                                            vector<int>& chambers,
                                                            vector<int>& rolls,
                                                            const GEMGeometry& geo) {
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonGEMDetLayerGeometryBuilder";
  ForwardDetLayer* result = nullptr;
  vector<const ForwardDetRing*> frontRings, backRings;
  for (std::vector<int>::iterator ring = rings.begin(); ring != rings.end(); ring++) {
    for (vector<int>::iterator roll = rolls.begin(); roll != rolls.end(); roll++) {
      vector<const GeomDet*> frontDets, backDets;

      for (std::vector<int>::iterator chamber = chambers.begin(); chamber < chambers.end(); chamber++) {
        GEMDetId gemId(endcap, (*ring), station, layer, (*chamber), (*roll));

        const GeomDet* geomDet = geo.idToDet(gemId);

        if (geomDet != nullptr) {
          // ME0s do not currently have an arrangement of which are front and which are back, going to always return true
          bool isInFront = (station == GEMDetId::minStationId0) or isFront(gemId);
          if (isInFront) {
            frontDets.push_back(geomDet);
          } else {
            backDets.push_back(geomDet);
          }
          LogTrace(metname) << "get GEM Endcap roll " << gemId << (isInFront ? "front" : "back ")
                            << " at R=" << geomDet->position().perp() << ", phi=" << geomDet->position().phi()
                            << ", Z=" << geomDet->position().z();
        }
      }

      if (!frontDets.empty()) {
        precomputed_value_sort(frontDets.begin(), frontDets.end(), geomsort::DetPhi());
        frontRings.push_back(new MuDetRing(frontDets));
        LogTrace(metname) << "New front ring with " << frontDets.size()
                          << " chambers at z=" << frontRings.back()->position().z();
      }
      if (!backDets.empty()) {
        precomputed_value_sort(backDets.begin(), backDets.end(), geomsort::DetPhi());
        backRings.push_back(new MuDetRing(backDets));
        LogTrace(metname) << "New back ring with " << backDets.size()
                          << " chambers at z=" << backRings.back()->position().z();
      }
    }
  }

  // How should they be sorted?
  //    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractZ<GeometricSearchDet,float>());
  if (!backRings.empty() && !frontRings.empty() && station != GEMDetId::minStationId0) {
    result = new MuRingForwardDoubleLayer(frontRings, backRings);
  } else if (!frontRings.empty() && station == GEMDetId::minStationId0) {
    result = new MuRingForwardLayer(frontRings);
  } else {
    result = nullptr;
  }
  if (result != nullptr) {
    LogTrace(metname) << "New MuRingForwardLayer with " << frontRings.size() << " and " << backRings.size()
                      << " rings, at Z " << result->position().z() << " R1: " << result->specificSurface().innerRadius()
                      << " R2: " << result->specificSurface().outerRadius();
  }
  return result;
}

bool MuonGEMDetLayerGeometryBuilder::isFront(const GEMDetId& gemId) {
  bool result = false;
  int chamber = gemId.chamber();

  if (chamber % 2 == 0)
    result = !result;

  return result;
}

MuDetRing* MuonGEMDetLayerGeometryBuilder::makeDetRing(vector<const GeomDet*>& geomDets) {
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonGEMDetLayerGeometryBuilder";

  precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
  MuDetRing* result = new MuDetRing(geomDets);
  LogTrace(metname) << "New MuDetRing with " << geomDets.size() << " chambers at z=" << result->position().z()
                    << " R1: " << result->specificSurface().innerRadius()
                    << " R2: " << result->specificSurface().outerRadius();
  return result;
}
