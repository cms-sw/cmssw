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

static bool adjustRZrangeFromBending(Range &rzRange,
        const PixelRecoLineRZ &line,
        const PixelRecoPointRZ &point2,
        const ThirdHitPredictionFromCircle &predictionRPhi,
        const TrackingRegion& region,
        const Range &zrRange,
        bool barrel)
{
  float (PixelRecoLineRZ::*xfrm)(float) const =
                barrel ? &PixelRecoLineRZ::zAtR : &PixelRecoLineRZ::rAtZ;
  float rBound = region.originRBound();

  Range tIP = predictionRPhi.transverseIP();
  Range sIP(abs(tIP.first), abs(tIP.second));
  sIP.sort();
  if (tIP.first * tIP.second <= 0.)
    sIP.first = 0.;
  else if (sIP.first > rBound)
    return false;
  sIP.second = min(sIP.second, rBound);

  PixelRecoLineRZ line1(line.origin(), point2, sIP.first);
  PixelRecoLineRZ line2(line.origin(), point2, sIP.second);
  double z[4] = {
    (line1.*xfrm)(zrRange.first),
    (line2.*xfrm)(zrRange.first),
    (line1.*xfrm)(zrRange.second),
    (line2.*xfrm)(zrRange.second)
  };
  sort(z, z + 4);
  Range orig((line.*xfrm)(zrRange.first), (line.*xfrm)(zrRange.second));
  orig.sort();

  rzRange.first += min(z[0] - orig.first, z[3] - orig.second);
  rzRange.second += max(z[0] - orig.first, z[3] - orig.second);
  return true;
}

static bool intersect(Range &range, const ThirdHitRZPrediction::Range &second)
{
  if (range.first > second.max() ||
      range.second < second.min())
    return false;
  if (range.first < second.min())
    range.first = second.min();
  if (range.second > second.max())
    range.second = second.max();
  return true;
}


void PixelTripletLargeTipGenerator::hitTriplets( 
    const TrackingRegion& region, 
    OrderedHitTriplets & result,
    const edm::Event & ev,
    const edm::EventSetup& es)
{
  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  thePairGenerator->hitPairs(region,pairs,ev,es);

  if (pairs.empty()) return;

  int size = theLayers.size();

  const LayerHitMap **thirdHitMap = new const LayerHitMap* [size];
  for (int il=0; il <=size-1; il++) {
     thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
  }

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  double curv = PixelRecoUtilities::curvature(1/region.ptMin(), es);

  for (ip = pairs.begin(); ip != pairs.end(); ip++) {
    const TrackingRecHit * h1 = (*ip).inner();
    const TrackingRecHit * h2 = (*ip).outer();
    GlobalPoint gp1 = tracker->idToDet( 
        h1->geographicalId())->surface().toGlobal(h1->localPosition());
    GlobalPoint gp2 = tracker->idToDet( 
        h2->geographicalId())->surface().toGlobal(h2->localPosition());

    PixelRecoLineRZ line(gp1, gp2);
    PixelRecoPointRZ point2(gp2.perp(), gp2.z());
    ThirdHitRZPrediction predictionRZ(line, extraHitRZtolerance);
    ThirdHitPredictionFromCircle predictionRPhi(gp1,gp2,curv,extraHitRPhitolerance);

    for (int il=0; il <=size-1; il++) {
      const DetLayer * layer = theLayers[il].detLayer();
      bool pixelLayer = layer->subDetector() == GeomDetEnumerators::PixelBarrel ||
                        layer->subDetector() == GeomDetEnumerators::PixelEndcap;
      bool barrelLayer = layer->location() == GeomDetEnumerators::barrel;

      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2, useMScat);

      ThirdHitRZPrediction::Range rzRange = predictionRZ(layer);
      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) { 
        float phi0 = (*ip).outer().phi();
        phiRange = Range(phi0-dphi, phi0+dphi);
      } else {
        Range radius;

        if (barrelLayer) {
          radius = predictionRZ.detRange();
          if (!intersect(rzRange, predictionRZ.detSize()))
            continue;
        } else {
          radius = rzRange;
          if (!intersect(radius, predictionRZ.detSize()))
            continue;
        }

        Range rPhi1 = predictionRPhi(radius.first);
        Range rPhi2 = predictionRPhi(radius.second);
        correction.correctRPhiRange(rPhi1);
        correction.correctRPhiRange(rPhi2);
        rPhi1.first  /= radius.first;
        rPhi1.second /= radius.first;
        rPhi2.first  /= radius.second;
        rPhi2.second /= radius.second;
        phiRange = mergePhiRanges(rPhi1, rPhi2);

        if (useBend && !adjustRZrangeFromBending(
                            rzRange, line, point2, predictionRPhi, region,
                            barrelLayer ? radius : predictionRZ.detRange(),
                            barrelLayer))
            continue;
      }

      LayerHitMapLoop thirdHits = 
          pixelLayer ? thirdHitMap[il]->loop(phiRange, rzRange)
                     : thirdHitMap[il]->loop();

      const SeedingHit * th;
      while ( (th = thirdHits.getHit()) ) {
         float p3_r = th->r();
         float p3_z = th->z();
         float p3_phi = th->phi();

         Range rangeRPhi = predictionRPhi(p3_r);
         correction.correctRPhiRange(rangeRPhi);
         if (!checkPhiInRange(p3_phi, rangeRPhi.first/p3_r, rangeRPhi.second/p3_r))
           continue;

         const TransientTrackingRecHit::ConstRecHitPointer& hit = *th;
         Basic2DVector<double> thc(hit->globalPosition().x(),
                                   hit->globalPosition().y());
         float tIP = min((float)predictionRPhi.transverseIP(thc),
                         region.originRBound());

         PixelRecoLineRZ updatedLine(line.origin(), point2, tIP);
         ThirdHitRZPrediction::Range rangeRZ =
                         predictionRZ(barrelLayer ? p3_r : p3_z, updatedLine);
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

