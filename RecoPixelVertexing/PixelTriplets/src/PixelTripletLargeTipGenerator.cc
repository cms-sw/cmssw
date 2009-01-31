#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletLargeTipGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMap.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/src/ThirdHitCorrection.h"

#include <algorithm>
#include <iostream>
#include <cmath>

using namespace std;
using namespace ctfseeding;

typedef PixelRecoRange<float> Range;

typedef ThirdHitPredictionFromCircle::HelixRZ HelixRZ;

static const double nSigmaRZ = 3.4641016151377544; // sqrt(12.)
static const double nSigmaPhi = 3.;

PixelTripletLargeTipGenerator::PixelTripletLargeTipGenerator(const edm::ParameterSet& cfg)
    : thePairGenerator(0),
      theLayerCache(0),
      useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
      extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
      extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
      useMScat(cfg.getParameter<bool>("useMultScattering")),
      useBend(cfg.getParameter<bool>("useBending"))
{
  if (useFixedPreFiltering)
    dphi = cfg.getParameter<double>("phiPreFiltering");
}

void PixelTripletLargeTipGenerator::init(const HitPairGenerator & pairs,
      const std::vector<SeedingLayer> & layers,
      LayerCacheType* layerCache)
{
  thePairGenerator = pairs.clone();
  theLayers = layers;
  theLayerCache = layerCache;
}

static bool intersect(Range &range, const Range &second)
{
  if (range.first >= second.max() || range.second <= second.min())
    return false;
  if (range.first < second.min())
    range.first = second.min();
  if (range.second > second.max())
    range.second = second.max();
  return range.first < range.second;
}

void PixelTripletLargeTipGenerator::hitTriplets( 
    const TrackingRegion& region, 
    OrderedHitTriplets & result,
    const edm::Event & ev,
    const edm::EventSetup& es)
{
  OrderedHitPairs pairs;
  pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  thePairGenerator->hitPairs(region,pairs,ev,es);

  if (pairs.empty())
    return;

  int size = theLayers.size();

  const LayerHitMap **thirdHitMap = new const LayerHitMap*[size];
  for(int il = 0; il < size; il++)
     thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  double curv = PixelRecoUtilities::curvature(1. / region.ptMin(), es);

  for(ip = pairs.begin(); ip != pairs.end(); ip++) {
    const TrackingRecHit * h1 = (*ip).inner();
    const TrackingRecHit * h2 = (*ip).outer();
    GlobalPoint gp1 = tracker->idToDet( 
        h1->geographicalId())->surface().toGlobal(h1->localPosition());
    GlobalPoint gp2 = tracker->idToDet( 
        h2->geographicalId())->surface().toGlobal(h2->localPosition());

    PixelRecoLineRZ line(gp1, gp2);
    PixelRecoPointRZ point2(gp2.perp(), gp2.z());
    ThirdHitPredictionFromCircle predictionRPhi(gp1, gp2, extraHitRPhitolerance);
    ThirdHitRZPrediction<PixelRecoLineRZ> predLineRZ(&line, extraHitRZtolerance);

    Range generalCurvature = predictionRPhi.curvature(region.originRBound());
    if (!intersect(generalCurvature, Range(-curv, curv)))
      continue;

    for (int il=0; il <=size-1; il++) {
      const DetLayer * layer = theLayers[il].detLayer();
      bool pixelLayer = layer->subDetector() == GeomDetEnumerators::PixelBarrel ||
                        layer->subDetector() == GeomDetEnumerators::PixelEndcap;
      bool barrelLayer = layer->location() == GeomDetEnumerators::barrel;

      Range curvature = generalCurvature;
      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2, useMScat);

      predLineRZ.initLayer(layer);

      HelixRZ helix;
      ThirdHitRZPrediction<HelixRZ> predHelixRZ;
      Range rzRange;
      if (useBend) {
        // For the barrel region:
        // swiping the helix passing through the two points across from
        // negative to positive bending, can give us a sort of U-shaped
        // projection onto the phi-z (barrel) or r-z plane (forward)
        // so we checking minimum/maximum of all three possible extrema
        // 
        // For the endcap region:
        // Checking minimum/maximum radius of the helix projection
        // onto an endcap plane, here we have to guard against
        // looping tracks, when phi(delta z) gets out of control.
        // HelixRZ::rAtZ should not follow looping tracks, but clamp
        // to the minimum reachable r with the next-best lower |curvature|.
        // So same procedure as for the barrel region can be applied.
        //
        // In order to avoid looking for potential looping tracks at all
        // we also clamp the allowed curvature range for this layer,
        // and potentially fail the layer entirely

        if (!barrelLayer) {
          Range z3s = predLineRZ.detRange();
          double z3 = z3s.first < 0 ? max(z3s.first, z3s.second)
                                    : min(z3s.first, z3s.second);
          double maxCurvature = HelixRZ::maxCurvature(&predictionRPhi,
                                                      gp1.z(), gp2.z(), z3);
          if (!intersect(curvature, Range(-maxCurvature, maxCurvature)))
            continue;
        }

        helix = HelixRZ(&predictionRPhi, gp1.z(), gp2.z(), curvature.first);
        HelixRZ helix2(&predictionRPhi, gp1.z(), gp2.z(), curvature.second);

        predHelixRZ = ThirdHitRZPrediction<HelixRZ>(&helix,
                                                    extraHitRZtolerance);
        ThirdHitRZPrediction<HelixRZ> predHelix2RZ(&helix2,
                                                   extraHitRZtolerance);

        Range rzRanges[2] = { predHelixRZ(layer), predHelix2RZ(layer) };
        rzRange.first = min(rzRanges[0].first, rzRanges[1].first);
        rzRange.second = max(rzRanges[0].second, rzRanges[1].second);

        // if the allowed curvatures include a straight line,
        // this can give us another extremum for allowed r/z
        if (curvature.first * curvature.second < 0.0) {
          Range rzLineRange = predLineRZ();
          rzRange.first = min(rzRange.first, rzLineRange.first);
          rzRange.second = max(rzRange.second, rzLineRange.second);
        }
      } else
        rzRange = predLineRZ(layer);

      if (rzRange.first == rzRange.second)
        continue;

      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) { 
        float phi0 = (*ip).outer().phi();
        phiRange = Range(phi0 - dphi, phi0 + dphi);
      } else {
        Range radius;

        if (barrelLayer) {
          radius = predLineRZ.detRange();
          if (!intersect(rzRange, predLineRZ.detSize()))
            continue;
        } else {
          radius = rzRange;
          if (!intersect(radius, predLineRZ.detSize()))
            continue;
        }

        Range rPhi1 = predictionRPhi(curvature, radius.first);
        Range rPhi2 = predictionRPhi(curvature, radius.second);
        correction.correctRPhiRange(rPhi1);
        correction.correctRPhiRange(rPhi2);
        rPhi1.first  /= radius.first;
        rPhi1.second /= radius.first;
        rPhi2.first  /= radius.second;
        rPhi2.second /= radius.second;
        phiRange = mergePhiRanges(rPhi1, rPhi2);
      }

      LayerHitMapLoop thirdHits = 
          pixelLayer ? thirdHitMap[il]->loop(phiRange, rzRange)
                     : thirdHitMap[il]->loop();

      const SeedingHit * th;
      while ( (th = thirdHits.getHit()) ) {
         float p3_r = th->r();
         float p3_z = th->z();
         float p3_phi = th->phi();

         Range rangeRPhi = predictionRPhi(curvature, p3_r);
         correction.correctRPhiRange(rangeRPhi);
         if (!checkPhiInRange(p3_phi, rangeRPhi.first/p3_r, rangeRPhi.second/p3_r))
           continue;

         const TransientTrackingRecHit::ConstRecHitPointer& hit = *th;
         Basic2DVector<double> thc(hit->globalPosition().x(),
                                   hit->globalPosition().y());

         Range rangeRZ;
         if (useBend) {
           HelixRZ updatedHelix(&predictionRPhi, gp1.z(), gp2.z(),
           	                predictionRPhi.curvature(thc));
           rangeRZ = predHelixRZ(barrelLayer ? p3_r : p3_z, updatedHelix);
         } else {
           float tIP = predictionRPhi.transverseIP(thc);
           PixelRecoLineRZ updatedLine(line.origin(), point2, tIP);
           rangeRZ = predLineRZ(barrelLayer ? p3_r : p3_z, line);
         }
         correction.correctRZRange(rangeRZ);

         double err = nSigmaRZ * sqrt(barrelLayer
                 ? hit->globalPositionError().czz()
                 : hit->globalPositionError().rerr(hit->globalPosition()));
         rangeRZ.first -= err, rangeRZ.second += err;

         if (!rangeRZ.inside(barrelLayer ? p3_z : p3_r)) continue;

         result.push_back(OrderedHitTriplet((*ip).inner(), (*ip).outer(), *th)); 
      }
    }
  }
  delete [] thirdHitMap;
}

bool PixelTripletLargeTipGenerator::checkPhiInRange(float phi, float phi1, float phi2) const
{
  while (phi > phi2) phi -= 2*M_PI;
  while (phi < phi1) phi += 2*M_PI;
  return phi <= phi2;
}  

std::pair<float,float> PixelTripletLargeTipGenerator::mergePhiRanges(
          const std::pair<float,float>& r1, const std::pair<float,float>& r2) const
{
  float r2_min=r2.first;
  float r2_max=r2.second;
  while (r1.first-r2_min > M_PI) { r2_min += 2*M_PI; r2_max += 2*M_PI;}
  while (r1.first-r2_min < -M_PI) { r2_min -= 2*M_PI;  r2_max -= 2*M_PI; }

  return std::make_pair(min(r1.first,r2_min),max(r1.second,r2_max));
}
