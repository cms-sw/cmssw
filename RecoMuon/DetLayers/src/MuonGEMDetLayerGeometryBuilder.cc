#include <RecoMuon/DetLayers/src/MuonGEMDetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h>
#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <RecoMuon/DetLayers/interface/MuDetRod.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonGEMDetLayerGeometryBuilder::~MuonGEMDetLayerGeometryBuilder() {}

// Builds the forward (first) and backward (second) layers
// Builds etaPartitions (for rechits)
pair<vector<DetLayer*>, vector<DetLayer*> > MuonGEMDetLayerGeometryBuilder::buildEndcapLayers(const GEMGeometry& geo) {
  vector<DetLayer*> endcapLayers[2];

  for (auto st : geo.stations()) {
    const int maxLayerId = (st->station() == GEMDetId::minStationId0) ? GEMDetId::maxLayerId0 : GEMDetId::maxLayerId;
    for (int layer = GEMDetId::minLayerId + 1; layer <= maxLayerId; ++layer) {
      ForwardDetLayer* forwardLayer = nullptr;
      vector<const ForwardDetRing*> frontRings, backRings;

      for (int roll = GEMDetId::minRollId + 1; roll <= GEMDetId::maxRollId; ++roll) {
        vector<const GeomDet*> frontDets, backDets;

        for (auto sc : st->superChambers()) {
          auto ch = sc->chamber(layer);
          if (ch == nullptr)
            continue;

          auto etaP = ch->etaPartition(roll);
          if (etaP == nullptr)
            continue;

          bool isInFront = isFront(etaP->id());
          if (isInFront) {
            frontDets.push_back(etaP);
          } else {
            backDets.push_back(etaP);
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

      if (!frontRings.empty()) {
        if (st->station() == GEMDetId::minStationId0)
          forwardLayer = new MuRingForwardLayer(frontRings);
        else if (!backRings.empty())
          forwardLayer = new MuRingForwardDoubleLayer(frontRings, backRings);
      }

      if (forwardLayer != nullptr) {
        LogTrace(metname) << "New MuRingForwardLayer with " << frontRings.size() << " and " << backRings.size()
                          << " rings, at Z " << forwardLayer->position().z()
                          << " R1: " << forwardLayer->specificSurface().innerRadius()
                          << " R2: " << forwardLayer->specificSurface().outerRadius();

        int iendcap = (st->region() == 1) ? 0 : 1;
        endcapLayers[iendcap].push_back(forwardLayer);
      }
    }
  }

  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(endcapLayers[0], endcapLayers[1]);
  return res_pair;
}

bool MuonGEMDetLayerGeometryBuilder::isFront(const GEMDetId& gemId) {
  // ME0s do not currently have an arrangement of which are front and which are back, going to always return true
  if (gemId.station() == GEMDetId::minStationId0)
    return true;

  if (gemId.chamber() % 2 == 0)
    return true;

  return false;
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
