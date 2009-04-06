#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletNoTipGenerator.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/src/ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvLine.h"

#include <iostream>
using pixelrecoutilities::LongitudinalBendingCorrection;
typedef PixelRecoRange<float> Range;

using namespace std;
using namespace ctfseeding;

PixelTripletNoTipGenerator:: PixelTripletNoTipGenerator(const edm::ParameterSet& cfg)
    : thePairGenerator(0),
      theLayerCache(0),
      extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
      extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
      useMScat(cfg.getParameter<bool>("useMultScattering")),
      useBend(cfg.getParameter<bool>("useBending")),
      theBeamSpotTag(cfg.getParameter<edm::InputTag>("beamSpot"))
{ }

void PixelTripletNoTipGenerator::init( const HitPairGenerator & pairs,
      const std::vector<SeedingLayer> & layers,
      LayerCacheType* layerCache)
{
  thePairGenerator = pairs.clone();
  theLayers = layers;
  theLayerCache = layerCache;
}

void PixelTripletNoTipGenerator::hitTriplets(
    const TrackingRegion& region,
    OrderedHitTriplets & result,
    const edm::Event & ev,
    const edm::EventSetup& es)
{
  std::cout << "KUKU **" << std::endl;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByLabel( theBeamSpotTag, bsHandle);
    if(!bsHandle.isValid()) return;

    const reco::BeamSpot & bs = *bsHandle;
    GlobalPoint bsPoint(bs.x0(), bs.y0(), bs.z0());

//      GlobalPoint origin(bs.x0(), bs.y0(), bs.z0());
//      std::cout <<"BeamSpot: " << origin << "sigma z: "<< bs.sigmaZ() <<" error xy:"<< bs.BeamWidth()<<std::endl;
//      std::cout << bs << std::endl;
//      result.push_back( new GlobalTrackingRegion(
//          thePtMin, origin, theOriginRadius, theNSigmaZ*bs.sigmaZ(), thePrecise));

  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  thePairGenerator->hitPairs(region,pairs,ev,es);

  if (pairs.size() ==0) return;

  int size = theLayers.size();

  const RecHitsSortedInPhi **thirdHitMap = new const RecHitsSortedInPhi*[size];
  for (int il=0; il <=size-1; il++) {
     thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
  }

  typedef RecHitsSortedInPhi::Hit Hit;
  for (ip = pairs.begin(); ip != pairs.end(); ip++) {
    GlobalPoint p1((*ip).inner()->globalPosition().x()-bsPoint.x(), (*ip).inner()->globalPosition().y()-bsPoint.y(), (*ip).inner()->globalPosition().z()-bsPoint.z() );
    GlobalPoint p2((*ip).outer()->globalPosition().x()-bsPoint.x(), (*ip).outer()->globalPosition().y()-bsPoint.y(), (*ip).outer()->globalPosition().z()-bsPoint.z() );
    ThirdHitPredictionFromInvLine  predictionRPhi(p1,p2);
    GlobalPoint center = predictionRPhi.center();
    std::cout << center.perp() << "pt = "<< 1/PixelRecoUtilities::inversePt(1./center.perp(), es) << std::endl;
    for (int il=0; il <=size-1; il++) {

      vector<Hit> thirdHits = thirdHitMap[il]->hits(); 
      typedef vector<Hit>::const_iterator IH;
      for (IH th=thirdHits.begin(), eh=thirdHits.end(); th < eh; ++th) {
         result.push_back( OrderedHitTriplet( (*ip).inner(), (*ip).outer(), *th));
      } 
    }
  }
  delete [] thirdHitMap;
}

bool PixelTripletNoTipGenerator::checkPhiInRange(float phi, float phi1, float phi2) const
{
  while (phi > phi2) phi -=  2*M_PI;
  while (phi < phi1) phi +=  2*M_PI;
  return (  (phi1 <= phi) && (phi <= phi2) );
}

std::pair<float,float> PixelTripletNoTipGenerator::mergePhiRanges(
    const std::pair<float,float>& r1, const std::pair<float,float>& r2) const
{
  float r2_min=r2.first;
  float r2_max=r2.second;
  while (r1.first-r2_min > M_PI) { r2_min += 2*M_PI; r2_max += 2*M_PI;}
  while (r1.first-r2_min < -M_PI) { r2_min -= 2*M_PI;  r2_max -= 2*M_PI; }

  return std::make_pair(min(r1.first,r2_min),max(r1.second,r2_max));
}


