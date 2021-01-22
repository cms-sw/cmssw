#include "RecoEgamma/EgammaElectronAlgos/interface/utils.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/ftsFromVertexToPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

using namespace reco;
using namespace std;

bool PixelHitMatcher::ForwardMeasurementEstimator::operator()(const GlobalPoint &vprim,
                                                              const TrajectoryStateOnSurface &absolute_ts,
                                                              const GlobalPoint &absolute_gp,
                                                              int charge) const {
  GlobalVector ts = absolute_ts.globalParameters().position() - vprim;
  GlobalVector gp = absolute_gp - vprim;

  float rDiff = gp.perp() - ts.perp();
  float rMin = theRMin;
  float rMax = theRMax;
  float myZ = gp.z();
  if ((std::abs(myZ) > 70.f) & (std::abs(myZ) < 170.f)) {
    rMin = theRMinI;
    rMax = theRMaxI;
  }

  if ((rDiff >= rMax) | (rDiff <= rMin))
    return false;

  float phiDiff = -charge * normalizedPhi(gp.barePhi() - ts.barePhi());

  return (phiDiff < thePhiMax) & (phiDiff > thePhiMin);
}

bool PixelHitMatcher::BarrelMeasurementEstimator::operator()(const GlobalPoint &vprim,
                                                             const TrajectoryStateOnSurface &absolute_ts,
                                                             const GlobalPoint &absolute_gp,
                                                             int charge) const {
  GlobalVector ts = absolute_ts.globalParameters().position() - vprim;
  GlobalVector gp = absolute_gp - vprim;

  float myZ = gp.z();
  float zDiff = myZ - ts.z();
  float myZmax = theZMax;
  float myZmin = theZMin;
  if ((std::abs(myZ) < 30.f) & (gp.perp() > 8.f)) {
    myZmax = 0.09f;
    myZmin = -0.09f;
  }

  if ((zDiff >= myZmax) | (zDiff <= myZmin))
    return false;

  float phiDiff = -charge * normalizedPhi(gp.barePhi() - ts.barePhi());

  return (phiDiff < thePhiMax) & (phiDiff > thePhiMin);
}

PixelHitMatcher::PixelHitMatcher(float phi1min,
                                 float phi1max,
                                 float phi2minB,
                                 float phi2maxB,
                                 float phi2minF,
                                 float phi2maxF,
                                 float z2maxB,
                                 float r2maxF,
                                 float rMaxI,
                                 bool useRecoVertex)
    :  //zmin1 and zmax1 are dummy at this moment, set from beamspot later
      meas1stBLayer{phi1min, phi1max, 0., 0.},
      meas2ndBLayer{phi2minB, phi2maxB, -z2maxB, z2maxB},
      meas1stFLayer{phi1min, phi1max, 0., 0., -rMaxI, rMaxI},
      meas2ndFLayer{phi2minF, phi2maxF, -r2maxF, r2maxF, -rMaxI, rMaxI},
      useRecoVertex_(useRecoVertex) {}

void PixelHitMatcher::set1stLayer(float dummyphi1min, float dummyphi1max) {
  meas1stBLayer.thePhiMin = dummyphi1min;
  meas1stBLayer.thePhiMax = dummyphi1max;
  meas1stFLayer.thePhiMin = dummyphi1min;
  meas1stFLayer.thePhiMax = dummyphi1max;
}

void PixelHitMatcher::set1stLayerZRange(float zmin1, float zmax1) {
  meas1stBLayer.theZMin = zmin1;
  meas1stBLayer.theZMax = zmax1;
  meas1stFLayer.theRMin = zmin1;
  meas1stFLayer.theRMax = zmax1;
}

void PixelHitMatcher::set2ndLayer(float dummyphi2minB, float dummyphi2maxB, float dummyphi2minF, float dummyphi2maxF) {
  meas2ndBLayer.thePhiMin = dummyphi2minB;
  meas2ndBLayer.thePhiMax = dummyphi2maxB;
  meas2ndFLayer.thePhiMin = dummyphi2minF;
  meas2ndFLayer.thePhiMax = dummyphi2maxF;
}

void PixelHitMatcher::setES(MagneticField const &magField, TrackerGeometry const &trackerGeometry) {
  theMagField = &magField;
  theTrackerGeometry = &trackerGeometry;
  constexpr float mass = .000511;  // electron propagation
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

  auto fts = trackingTools::ftsFromVertexToPoint(*theMagField, xmeas, vprim, energy, charge);
  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));

  std::vector<SeedWithInfo> result;
  unsigned int allSeedsSize = 0;
  for (auto const sc : seedsV)
    allSeedsSize += sc->size();

  IntGlobalPointPairUnorderedMap<TrajectoryStateOnSurface> mapTsos2Fast(allSeedsSize);

  auto ndets = theTrackerGeometry->dets().size();

  int iTsos[ndets];
  for (auto &i : iTsos)
    i = -1;
  std::vector<TrajectoryStateOnSurface> vTsos;
  vTsos.reserve(allSeedsSize);

  std::vector<GlobalPoint> hitGpMap;
  for (const auto seeds : seedsV) {
    for (const auto &seed : *seeds) {
      hitGpMap.clear();
      if (seed.nHits() > 9) {
        edm::LogWarning("GsfElectronAlgo|UnexpectedSeed") << "We cannot deal with seeds having more than 9 hits.";
        continue;
      }

      auto const &hits = seed.recHits();
      // cache the global points

      for (auto const &hit : hits) {
        hitGpMap.emplace_back(hit.globalPosition());
      }

      //iterate on the hits
      auto he = hits.end() - 1;
      for (auto it1 = hits.begin(); it1 < he; ++it1) {
        if (!it1->isValid())
          continue;
        auto idx1 = std::distance(hits.begin(), it1);
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
        bool est = id1.subdetId() % 2 ? meas1stBLayer(vprim, *tsos1, hit1Pos, charge)
                                      : meas1stFLayer(vprim, *tsos1, hit1Pos, charge);
        if (!est)
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
        auto fts2 = trackingTools::ftsFromVertexToPoint(*theMagField, hit1Pos, vertex, energy, charge);
        // now find the matching hit
        for (auto it2 = it1 + 1; it2 != hits.end(); ++it2) {
          if (!it2->isValid())
            continue;
          auto idx2 = std::distance(hits.begin(), it2);
          const DetId id2 = it2->geographicalId();
          const GeomDet *geomdet2 = it2->det();
          const auto det_key = std::make_pair(geomdet2->gdetIndex(), hit1Pos);
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
          bool est2 = id2.subdetId() % 2 ? meas2ndBLayer(vertex, *tsos2, hit2Pos, charge)
                                         : meas2ndFLayer(vertex, *tsos2, hit2Pos, charge);
          if (est2) {
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
