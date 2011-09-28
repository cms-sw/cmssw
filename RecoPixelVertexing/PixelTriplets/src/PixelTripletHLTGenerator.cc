#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletHLTGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/src/ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "DataFormats/GeometryVector/interface/Pi.h"

using pixelrecoutilities::LongitudinalBendingCorrection;
typedef PixelRecoRange<float> Range;

using namespace std;
using namespace ctfseeding;

PixelTripletHLTGenerator:: PixelTripletHLTGenerator(const edm::ParameterSet& cfg)
    : thePairGenerator(0),
      theLayerCache(0),
      useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
      extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
      extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
      useMScat(cfg.getParameter<bool>("useMultScattering")),
      useBend(cfg.getParameter<bool>("useBending"))
{
  theMaxElement=cfg.getParameter<unsigned int>("maxElement");
  dphi =  (useFixedPreFiltering) ?  cfg.getParameter<double>("phiPreFiltering") : 0;

  if (cfg.exists("SeedComparitorPSet")){
  edm::ParameterSet comparitorPSet =
    cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  theComparitor = (comparitorName == "none") ?
    0 :  SeedComparitorFactory::get()->create( comparitorName, comparitorPSet);
  }
  else
    theComparitor=0;
}

PixelTripletHLTGenerator::~PixelTripletHLTGenerator()
  
{ delete thePairGenerator;
  delete theComparitor;
}

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
  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  
  thePairGenerator->hitPairs(region,pairs,ev,es);

  if (pairs.empty()) return;

  int size = theLayers.size();

  typedef std::vector<ThirdHitRZPrediction<PixelRecoLineRZ> >  Preds;
  Preds preds(size);

  std::vector<const RecHitsSortedInPhi *> thirdHitMap(size);
  typedef RecHitsSortedInPhi::Hit Hit;
  vector<Hit> thirdHits;

  // fill the prediciton vetor
  for (int il=0; il!=size; ++il) {
     thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
     ThirdHitRZPrediction<PixelRecoLineRZ> & pred = preds[il];
     pred.initLayer(theLayers[il].detLayer());
     pred.initTolerance(extraHitRZtolerance);
  }


  double imppar = region.originRBound();
  double curv = PixelRecoUtilities::curvature(1/region.ptMin(), es);

  for (ip = pairs.begin(); ip != pairs.end(); ip++) {
  
    GlobalPoint gp1tmp = (*ip).inner()->globalPosition();
    GlobalPoint gp2tmp = (*ip).outer()->globalPosition();
    GlobalPoint gp1(gp1tmp.x()-region.origin().x(), gp1tmp.y()-region.origin().y(), gp1tmp.z());
    GlobalPoint gp2(gp2tmp.x()-region.origin().x(), gp2tmp.y()-region.origin().y(), gp2tmp.z());

    PixelRecoPointRZ point1(gp1.perp(), gp1.z());
    PixelRecoPointRZ point2(gp2.perp(), gp2.z());
    PixelRecoLineRZ  line(point1, point2);
    ThirdHitPredictionFromInvParabola predictionRPhi(gp1,gp2,imppar,curv,extraHitRPhitolerance);
    ThirdHitPredictionFromInvParabola predictionRPhitmp(gp1tmp,gp2tmp,imppar+region.origin().perp(),curv,extraHitRPhitolerance);


    for (int il=0; il!=size; ++il) {
      const DetLayer * layer = theLayers[il].detLayer();
//      bool pixelLayer = (    layer->subDetector() == GeomDetEnumerators::PixelBarrel 
//                          || layer->subDetector() == GeomDetEnumerators::PixelEndcap); 
      bool barrelLayer = (layer->location() == GeomDetEnumerators::barrel);

      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2, useMScat, useBend); 
      
      ThirdHitRZPrediction<PixelRecoLineRZ> & predictionRZ =  preds[il];

      predictionRZ.initPropagator(&line);
      Range rzRange = predictionRZ();

      correction.correctRZRange(rzRange);
      Range phiRange;
      if (useFixedPreFiltering) { 
        float phi0 = (*ip).outer()->globalPosition().phi();
        phiRange = Range(phi0-dphi,phi0+dphi);
      }
      else {
        Range radius;
        if (barrelLayer) {
          radius =  predictionRZ.detRange();
        } else {
          radius = Range(
              max(rzRange.min(), predictionRZ.detSize().min()),
              min(rzRange.max(), predictionRZ.detSize().max()) );
        }
        if (radius.empty()) continue;
        Range rPhi1m = predictionRPhitmp(radius.max(), -1);
        Range rPhi1p = predictionRPhitmp(radius.max(),  1);
        Range rPhi2m = predictionRPhitmp(radius.min(), -1);
        Range rPhi2p = predictionRPhitmp(radius.min(),  1);
        Range rPhi1 = rPhi1m.sum(rPhi1p);
        Range rPhi2 = rPhi2m.sum(rPhi2p);
        correction.correctRPhiRange(rPhi1);
        correction.correctRPhiRange(rPhi2);
        rPhi1.first  /= radius.max();
        rPhi1.second /= radius.max();
        rPhi2.first  /= radius.min();
        rPhi2.second /= radius.min();
        phiRange = mergePhiRanges(rPhi1,rPhi2);
      }
      
//      LayerHitMapLoop thirdHits = 
//          pixelLayer ? thirdHitMap[il]->loop(phiRange, rzRange) : 
//          thirdHitMap[il]->loop();

      thirdHits.clear();
      thirdHitMap[il]->hits(phiRange.min(),phiRange.max(), thirdHits);
  
      static float nSigmaRZ = std::sqrt(12.f);
      static float nSigmaPhi = 3.f;
   
      typedef vector<Hit>::const_iterator IH;
      for (IH th=thirdHits.begin(), eh=thirdHits.end(); th !=eh; ++th) {

        if (theMaxElement!=0 && result.size() >= theMaxElement){
	  result.clear();
	  edm::LogError("TooManyTriplets")<<" number of triples exceed maximum. no triplets produced.";
	  return;
	}
        const Hit& hit = (*th);
        GlobalPoint point(hit->globalPosition().x()-region.origin().x(),
                          hit->globalPosition().y()-region.origin().y(),
                          hit->globalPosition().z() ); 
        float p3_r = point.perp();
        float p3_z = point.z();
        float p3_phi = point.phi();
 
        if (barrelLayer) {
          Range allowedZ = predictionRZ(p3_r);
          correction.correctRZRange(allowedZ);

          float zErr = nSigmaRZ * std::sqrt(float(hit->globalPositionError().czz()));
          Range hitRange(p3_z-zErr, p3_z+zErr);
          Range crossingRange = allowedZ.intersection(hitRange);
          if (crossingRange.empty())  continue;
        } else {
          Range allowedR = predictionRZ(p3_z);
          correction.correctRZRange(allowedR); 
          float rErr = nSigmaRZ * std::sqrt(float(hit->globalPositionError().rerr( hit->globalPosition())));
          Range hitRange(p3_r-rErr, p3_r+rErr);
          Range crossingRange = allowedR.intersection(hitRange);
          if (crossingRange.empty())  continue;
        }

	float phiErr = nSigmaPhi*std::sqrt(float(hit->globalPositionError().phierr(hit->globalPosition())));
        for (int icharge=-1; icharge <=1; icharge+=2) {
          Range rangeRPhi = predictionRPhi(p3_r, icharge);
          correction.correctRPhiRange(rangeRPhi);
          if (checkPhiInRange(p3_phi, rangeRPhi.first/p3_r-phiErr, rangeRPhi.second/p3_r+phiErr)) {
	    // insert here check with comparitor
	    OrderedHitTriplet hittriplet( (*ip).inner(), (*ip).outer(), hit);
	    if(!theComparitor  || theComparitor->compatible(hittriplet,es) ) {
	      result.push_back( hittriplet );
	    } else {
	      LogDebug("RejectedTriplet") << "rejected triplet from comparitor "
					  << hittriplet.outer()->globalPosition().x() << " "
					  << hittriplet.outer()->globalPosition().y() << " "
					  << hittriplet.outer()->globalPosition().z();
	    }
	    break;
          } 
        }
      } 
    }
  }

}

bool PixelTripletHLTGenerator::checkPhiInRange(float phi, float phi1, float phi2) const
{
  while (phi > phi2) phi -=  Geom::ftwoPi();
  while (phi < phi1) phi +=  Geom::ftwoPi();
  return (  (phi1 <= phi) && (phi <= phi2) );
}  

std::pair<float,float> PixelTripletHLTGenerator::mergePhiRanges(
    const std::pair<float,float>& r1, const std::pair<float,float>& r2) const 
{
  float r2_min=r2.first;
  float r2_max=r2.second;
  while (r1.first-r2_min > Geom::fpi()) { r2_min += Geom::ftwoPi(); r2_max += Geom::ftwoPi();}
  while (r1.first-r2_min < -Geom::fpi()) { r2_min -= Geom::ftwoPi();  r2_max -= Geom::ftwoPi(); }
  
  return std::make_pair(min(r1.first,r2_min),max(r1.second,r2_max));
}
