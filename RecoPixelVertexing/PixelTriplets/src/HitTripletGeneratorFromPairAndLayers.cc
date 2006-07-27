#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMap.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"



void HitTripletGeneratorFromPairAndLayers::hitTriplets( 
    const TrackingRegion& region, 
    OrderedHitTriplets & result,
    const edm::EventSetup& iSetup)
{

  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  thePairGenerator->hitPairs(region,pairs,iSetup);

  if (pairs.size() ==0) return;

  int size = theLayers.size();

  const LayerHitMap **thirdHitMap = new const LayerHitMap* [size];
  for (int il=0; il <=size-1; il++) {
     thirdHitMap[il] = &theLayerCache(theLayers[il], region, iSetup);
  }

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  double imppar = region.originRBound();;
  double curv = PixelRecoUtilities::curvature(1/region.ptMin(), iSetup);

  for (ip = pairs.begin(); ip != pairs.end(); ip++) {
    const TrackingRecHit * h1 = (*ip).inner();
    const TrackingRecHit * h2 = (*ip).outer();
    GlobalPoint gp1 = tracker->idToDet( 
        h1->geographicalId())->surface().toGlobal(h1->localPosition());
    GlobalPoint gp2 = tracker->idToDet( 
        h2->geographicalId())->surface().toGlobal(h2->localPosition());

    ThirdHitRZPrediction predictionRZ(gp1,gp2);
    TkHitPairsCachedHit p2((*ip).outer(), iSetup);
    float phi0 = p2.phi();
    float dphi = 0.03;
    PixelRecoRange<float> phiRange(phi0-dphi,phi0+dphi);
    for (int il=0; il <=size-1; il++) {
      const LayerWithHits * layerwithhits = theLayers[il];
      const DetLayer * layer = layerwithhits->layer();
      bool barrelLayer = (layer->location() == GeomDetEnumerators::barrel);
      ThirdHitRZPrediction::Range rzRange = predictionRZ(layer);
      LayerHitMapLoop thirdHits = thirdHitMap[il]->loop(phiRange, rzRange);

      const TkHitPairsCachedHit * th;
      while ( (th = thirdHits.getHit()) ) {
         //GlobalPoint p3 = RecHit(*th).globalPosition();
         float p3_r = th->r();
         float p3_z = th->z();
         float p3_phi = th->phi();


         if (barrelLayer) {
           ThirdHitRZPrediction::Range rangeZ = predictionRZ(p3_r);
           if (! rangeZ.inside(p3_z) ) continue;
         } else {
           ThirdHitRZPrediction::Range rangeR = predictionRZ(p3_z);
           if (! rangeR.inside(p3_r) ) continue;
         }

         ThirdHitPredictionFromInvParabola predictionRPhi(gp1,gp2);

         double dist = predictionRPhi.isCompatible(GlobalPoint(p3_r*cos(p3_phi), p3_r*sin(p3_phi), p3_z), imppar, curv);
         if (fabs(dist) > 0.025) continue;

         const TrackingRecHit * h2 = (*ip).outer();
         const TrackingRecHit * h3 = th->RecHit();
         result.push_back( OrderedHitTriplet((*ip).inner(), h2, h3)); 
      }
    }
  }
  delete [] thirdHitMap;
}
