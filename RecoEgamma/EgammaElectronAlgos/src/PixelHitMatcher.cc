#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <typeinfo>
#include <bitset>

using namespace reco;
using namespace std;

PixelHitMatcher::PixelHitMatcher(float phi1min,
                                 float phi1max,
                                 float phi2minB,
                                 float phi2maxB,
                                 float phi2minF,
                                 float phi2maxF,
                                 float z2minB,
                                 float z2maxB,
                                 float r2minF,
                                 float r2maxF,
                                 float rMinI,
                                 float rMaxI,
                                 bool useRecoVertex)
    :  //zmin1 and zmax1 are dummy at this moment, set from beamspot later
      meas1stBLayer(phi1min, phi1max, 0., 0.),
      meas2ndBLayer(phi2minB, phi2maxB, z2minB, z2maxB),
      meas1stFLayer(phi1min, phi1max, 0., 0.),
      meas2ndFLayer(phi2minF, phi2maxF, r2minF, r2maxF),
      prop1stLayer(nullptr),
      prop2ndLayer(nullptr),
      useRecoVertex_(useRecoVertex) {
  meas1stFLayer.setRRangeI(rMinI, rMaxI);
  meas2ndFLayer.setRRangeI(rMinI, rMaxI);
}

void PixelHitMatcher::set1stLayer(float dummyphi1min, float dummyphi1max) {
  meas1stBLayer.setPhiRange(dummyphi1min, dummyphi1max);
  meas1stFLayer.setPhiRange(dummyphi1min, dummyphi1max);
}

void PixelHitMatcher::set1stLayerZRange(float zmin1, float zmax1) {
  meas1stBLayer.setZRange(zmin1, zmax1);
  meas1stFLayer.setRRange(zmin1, zmax1);
}

void PixelHitMatcher::set2ndLayer(float dummyphi2minB, float dummyphi2maxB, float dummyphi2minF, float dummyphi2maxF) {
  meas2ndBLayer.setPhiRange(dummyphi2minB, dummyphi2maxB);
  meas2ndFLayer.setPhiRange(dummyphi2minF, dummyphi2maxF);
}

void PixelHitMatcher::setES(const MagneticField *magField, const TrackerGeometry *trackerGeometry) {
  theMagField = magField;
  theTrackerGeometry = trackerGeometry;
  float mass = .000511;  // electron propagation
  prop1stLayer = std::make_unique<PropagatorWithMaterial>(oppositeToMomentum, mass, theMagField);
  prop2ndLayer = std::make_unique<PropagatorWithMaterial>(alongMomentum, mass, theMagField);
}

std::vector<SeedWithInfo> PixelHitMatcher::operator()(const std::vector<const TrajectorySeedCollection *> &seedsV,
                                                      const GlobalPoint &xmeas,
                                                      const GlobalPoint &vprim,
                                                      float energy,
                                                      int charge) const {
  auto xmeas_r = xmeas.perp();

  const float phicut = std::cos(2.5);

  FreeTrajectoryState fts = FTSFromVertexToPointFactory::get(*theMagField, xmeas, vprim, energy, charge);
  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));

  std::vector<SeedWithInfo> result;
  unsigned int allSeedsSize = 0;
  for (auto const sc : seedsV)
    allSeedsSize += sc->size();

  std::unordered_map<std::pair<const GeomDet *, GlobalPoint>, TrajectoryStateOnSurface> mapTsos2Fast;
  mapTsos2Fast.reserve(allSeedsSize);

  auto ndets = theTrackerGeometry->dets().size();

  int iTsos[ndets];
  for (auto &i : iTsos)
    i = -1;
  std::vector<TrajectoryStateOnSurface> vTsos;
  vTsos.reserve(allSeedsSize);

  for (const auto seeds : seedsV) {
    for (const auto &seed : *seeds) {
      std::vector<GlobalPoint> hitGpMap;
      if (seed.nHits() > 9) {
        edm::LogWarning("GsfElectronAlgo|UnexpectedSeed") << "We cannot deal with seeds having more than 9 hits.";
        continue;
      }

      const TrajectorySeed::range &hits = seed.recHits();
      // cache the global points

      for (auto it = hits.first; it != hits.second; ++it) {
        hitGpMap.emplace_back(it->globalPosition());
      }

      //iterate on the hits
      auto he = hits.second - 1;
      for (auto it1 = hits.first; it1 < he; ++it1) {
        if (!it1->isValid())
          continue;
        auto idx1 = std::distance(hits.first, it1);
        const DetId id1 = it1->geographicalId();
        const GeomDet *geomdet1 = it1->det();

        auto ix1 = geomdet1->gdetIndex();

        /*  VI: this generates regression (other cut is just in phi). in my opinion it is safe and makes sense
            auto away = geomdet1->position().basicVector().dot(xmeas.basicVector()) <0;
            if (away) continue;
        */

        const GlobalPoint &hit1Pos = hitGpMap[idx1];
        auto dt = hit1Pos.x() * xmeas.x() + hit1Pos.y() * xmeas.y();
        if (dt < 0)
          continue;
        if (dt < phicut * (xmeas_r * hit1Pos.perp()))
          continue;

        if (iTsos[ix1] < 0) {
          iTsos[ix1] = vTsos.size();
          vTsos.push_back(prop1stLayer->propagate(tsos, geomdet1->surface()));
        }
        auto tsos1 = &vTsos[iTsos[ix1]];

        if (!tsos1->isValid())
          continue;
        std::pair<bool, double> est = (id1.subdetId() % 2 ? meas1stBLayer.estimate(vprim, *tsos1, hit1Pos)
                                                          : meas1stFLayer.estimate(vprim, *tsos1, hit1Pos));
        if (!est.first)
          continue;
        EleRelPointPair pp1(hit1Pos, tsos1->globalParameters().position(), vprim);
        const math::XYZPoint relHit1Pos(hit1Pos - vprim), relTSOSPos(tsos1->globalParameters().position() - vprim);
        const int subDet1 = id1.subdetId();
        const float dRz1 = (id1.subdetId() % 2 ? pp1.dZ() : pp1.dPerp());
        const float dPhi1 = pp1.dPhi();
        // setup our vertex
        double zVertex;
        if (!useRecoVertex_) {
          // we don't know the z vertex position, get it from linear extrapolation
          // compute the z vertex from the cluster point and the found pixel hit
          const double pxHit1z = hit1Pos.z();
          const double pxHit1x = hit1Pos.x();
          const double pxHit1y = hit1Pos.y();
          const double r1diff =
              std::sqrt((pxHit1x - vprim.x()) * (pxHit1x - vprim.x()) + (pxHit1y - vprim.y()) * (pxHit1y - vprim.y()));
          const double r2diff =
              std::sqrt((xmeas.x() - pxHit1x) * (xmeas.x() - pxHit1x) + (xmeas.y() - pxHit1y) * (xmeas.y() - pxHit1y));
          zVertex = pxHit1z - r1diff * (xmeas.z() - pxHit1z) / r2diff;
        } else {
          // here use rather the reco vertex z position
          zVertex = vprim.z();
        }
        GlobalPoint vertex(vprim.x(), vprim.y(), zVertex);
        FreeTrajectoryState fts2 = FTSFromVertexToPointFactory::get(*theMagField, hit1Pos, vertex, energy, charge);
        // now find the matching hit
        for (auto it2 = it1 + 1; it2 != hits.second; ++it2) {
          if (!it2->isValid())
            continue;
          auto idx2 = std::distance(hits.first, it2);
          const DetId id2 = it2->geographicalId();
          const GeomDet *geomdet2 = it2->det();
          const std::pair<const GeomDet *, GlobalPoint> det_key(geomdet2, hit1Pos);
          const TrajectoryStateOnSurface *tsos2;
          auto tsos2_itr = mapTsos2Fast.find(det_key);
          if (tsos2_itr != mapTsos2Fast.end()) {
            tsos2 = &(tsos2_itr->second);
          } else {
            auto empl_result = mapTsos2Fast.emplace(det_key, prop2ndLayer->propagate(fts2, geomdet2->surface()));
            tsos2 = &(empl_result.first->second);
          }
          if (!tsos2->isValid())
            continue;
          const GlobalPoint &hit2Pos = hitGpMap[idx2];
          std::pair<bool, double> est2 = (id2.subdetId() % 2 ? meas2ndBLayer.estimate(vertex, *tsos2, hit2Pos)
                                                             : meas2ndFLayer.estimate(vertex, *tsos2, hit2Pos));
          if (est2.first) {
            EleRelPointPair pp2(hit2Pos, tsos2->globalParameters().position(), vertex);
            const int subDet2 = id2.subdetId();
            const float dRz2 = (subDet2 % 2 == 1) ? pp2.dZ() : pp2.dPerp();
            const float dPhi2 = pp2.dPhi();
            const unsigned char hitsMask = (1 << idx1) | (1 << idx2);
            result.push_back({seed, hitsMask, subDet2, dRz2, dPhi2, subDet1, dRz1, dPhi1});
          }
        }  // inner loop on hits
      }    // outer loop on hits
    }      // loop on seeds
  }        //loop on vector of seeds

  return result;
}
