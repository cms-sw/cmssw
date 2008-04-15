#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletHLTGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMap.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/src/ThirdHitCorrection.h"
#include <iostream>

#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

using pixelrecoutilities::LongitudinalBendingCorrection;
typedef PixelRecoRange<float> Range;
using namespace std;
using namespace ctfseeding;

void PixelTripletHLTGenerator::init( const HitPairGenerator & pairs,
      const std::vector<SeedingLayer> & layers,
      LayerCacheType* layerCache)
{
  thePairGenerator = pairs.clone();
  theLayers = layers;
  theLayerCache = layerCache;
}

void PixelTripletHLTGenerator::hitTriplets( 
    const TrackingRegion& region, 
    OrderedHitTriplets & result,
    const edm::Event & ev,
    const edm::EventSetup& es)
{

//  GlobalPoint origin = region.origin();
//  GlobalVector direction = region.direction();

  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  thePairGenerator->hitPairs(region,pairs,ev,es);

  if (pairs.size() ==0) return;

  int size = theLayers.size();

  const LayerHitMap **thirdHitMap = new const LayerHitMap* [size];
  for (int il=0; il <=size-1; il++) {
     thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
  }

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  double imppar = region.originRBound();;
  double curv = PixelRecoUtilities::curvature(1/region.ptMin(), es);

  bool useFixedPreFiltering =  theConfig.getParameter<bool>("useFixedPreFiltering");
  float extraHitRZtolerance = theConfig.getParameter<double>("extraHitRZtolerance");
  float extraHitRPhitolerance = theConfig.getParameter<double>("extraHitRPhitolerance");
  bool  useMScat = theConfig.getParameter<bool>("useMultScattering");
  bool  useBend  = theConfig.getParameter<bool>("useBending");


  for (ip = pairs.begin(); ip != pairs.end(); ip++) {
//
    const TrackingRecHit * h1 = (*ip).inner();
    const TrackingRecHit * h2 = (*ip).outer();
    GlobalPoint gp1 = tracker->idToDet( 
        h1->geographicalId())->surface().toGlobal(h1->localPosition());
    GlobalPoint gp2 = tracker->idToDet( 
        h2->geographicalId())->surface().toGlobal(h2->localPosition());

    PixelRecoPointRZ point1(gp1.perp(), gp1.z());
    PixelRecoPointRZ point2(gp2.perp(), gp2.z());
    PixelRecoLineRZ  line(point1, point2);
    ThirdHitRZPrediction predictionRZ(gp1,gp2,extraHitRZtolerance);
    ThirdHitPredictionFromInvParabola predictionRPhi(gp1,gp2,imppar,curv,extraHitRPhitolerance);

    for (int il=0; il <=size-1; il++) {
      const DetLayer * layer = theLayers[il].detLayer();
      bool pixelLayer = (    layer->subDetector() == GeomDetEnumerators::PixelBarrel 
                          || layer->subDetector() == GeomDetEnumerators::PixelEndcap); 
      bool barrelLayer = (layer->location() == GeomDetEnumerators::barrel);

      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2, useMScat, useBend);
      
      predictionRZ.initLayer(layer);

      ThirdHitRZPrediction::Range rzRange = predictionRZ(layer);
      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) { 
        static float dphi = theConfig.getParameter<double>("phiPreFiltering");
        float phi0 = (*ip).outer().phi();
        phiRange = Range(phi0-dphi,phi0+dphi);
      }
      else {
        float radiusMax = barrelLayer ?  predictionRZ.detRange().max() : rzRange.max();
        float radiusMin = barrelLayer ?  predictionRZ.detRange().min() : rzRange.min();

        if (radiusMin < 0 || radiusMax < 0) continue;
        Range rPhi1m = predictionRPhi(radiusMax, -1);
        Range rPhi1p = predictionRPhi(radiusMax,  1);
        Range rPhi2m = predictionRPhi(radiusMin, -1);
        Range rPhi2p = predictionRPhi(radiusMin,  1);
        Range rPhi1 = rPhi1m.sum(rPhi1p);
        Range rPhi2 = rPhi2m.sum(rPhi2p);
        correction.correctRPhiRange(rPhi1);
        correction.correctRPhiRange(rPhi2);
        rPhi1.first  /= radiusMax;
        rPhi1.second /= radiusMax;
        rPhi2.first  /= radiusMin;
        rPhi2.second /= radiusMin;
        phiRange = mergePhiRanges(rPhi1,rPhi2);
      }
      
      LayerHitMapLoop thirdHits = 
          pixelLayer ? thirdHitMap[il]->loop(phiRange, rzRange) : 
          thirdHitMap[il]->loop();
  
      const SeedingHit * th;
      double nSigmaRZ = sqrt(12.);
      double nSigmaPhi = 3.;
   
      while ( (th = thirdHits.getHit()) ) {

        float p3_r = th->r();
        float p3_z = th->z();
        float p3_phi = th->phi();
 
        const TransientTrackingRecHit::ConstRecHitPointer& hit = (*th);
        if (barrelLayer) {
          ThirdHitRZPrediction::Range allowedZ = predictionRZ(p3_r);
          correction.correctRZRange(allowedZ);

          double zErr = nSigmaRZ * sqrt(hit->globalPositionError().czz());
          Range hitRange(p3_z-zErr, p3_z+zErr);
          Range crossingRange = allowedZ.intersection(hitRange);
          if (crossingRange.empty())  continue;
        } else {
          ThirdHitRZPrediction::Range allowedR = predictionRZ(p3_z);
          correction.correctRZRange(allowedR); 
          double rErr = nSigmaRZ * sqrt(hit->globalPositionError().rerr( hit->globalPosition()));
          Range hitRange(p3_r-rErr, p3_r+rErr);
          Range crossingRange = allowedR.intersection(hitRange);
          if (crossingRange.empty())  continue;
        }

        for (int icharge=-1; icharge <=1; icharge+=2) {
          Range rangeRPhi = predictionRPhi(p3_r, icharge);
          correction.correctRPhiRange(rangeRPhi);
          double phiErr = nSigmaPhi * sqrt(hit->globalPositionError().phierr(hit->globalPosition()));
          if (checkPhiInRange(p3_phi, rangeRPhi.first/p3_r-phiErr, rangeRPhi.second/p3_r+phiErr)) {
            result.push_back( OrderedHitTriplet( (*ip).inner(), (*ip).outer(), *th)); 
            break;
          } 
        }
      } 
    }
  }
  delete [] thirdHitMap;
}

bool PixelTripletHLTGenerator::checkPhiInRange(float phi, float phi1, float phi2) const
{
  while (phi > phi2) phi -=  2*M_PI;
  while (phi < phi1) phi +=  2*M_PI;
  return (  (phi1 <= phi) && (phi <= phi2) );
}  

std::pair<float,float> PixelTripletHLTGenerator::mergePhiRanges(
    const std::pair<float,float>& r1, const std::pair<float,float>& r2) const 
{
  float r2_min=r2.first;
  float r2_max=r2.second;
  while (r1.first-r2_min > M_PI) { r2_min += 2*M_PI; r2_max += 2*M_PI;}
  while (r1.first-r2_min < -M_PI) { r2_min -= 2*M_PI;  r2_max -= 2*M_PI; }
  
  return std::make_pair(min(r1.first,r2_min),max(r1.second,r2_max));
}
