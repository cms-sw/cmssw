#include "RecoPixelVertexing/PixelTriplets/plugins/PixelTripletLargeTipGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

#include "MatchedHitRZCorrectionFromBending.h"
//#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"
//#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h" //amend to point at your copy...
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>

using namespace std;
using namespace ctfseeding;

typedef PixelRecoRange<float> Range;

typedef ThirdHitPredictionFromCircle::HelixRZ HelixRZ;

namespace {
  struct LayerRZPredictions {
    ThirdHitRZPrediction<PixelRecoLineRZ> line;
    ThirdHitRZPrediction<HelixRZ> helix1, helix2;
    MatchedHitRZCorrectionFromBending rzPositionFixup;
  };
}

constexpr double nSigmaRZ = 3.4641016151377544; // sqrt(12.)
constexpr double nSigmaPhi = 3.;
static float fnSigmaRZ = std::sqrt(12.f);

PixelTripletLargeTipGenerator::PixelTripletLargeTipGenerator(const edm::ParameterSet& cfg)
  : thePairGenerator(0),
    theLayerCache(0),
    useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
    extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
    extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
    useMScat(cfg.getParameter<bool>("useMultScattering")),
    useBend(cfg.getParameter<bool>("useBending"))
{    theMaxElement=cfg.getParameter<unsigned int>("maxElement");
  if (useFixedPreFiltering)
    dphi = cfg.getParameter<double>("phiPreFiltering");
}

void PixelTripletLargeTipGenerator::init(const HitPairGenerator & pairs,
					 const std::vector<SeedingLayer> &layers,
					 LayerCacheType *layerCache)
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

void PixelTripletLargeTipGenerator::hitTriplets(const TrackingRegion& region, 
						OrderedHitTriplets & result,
						const edm::Event & ev,
						const edm::EventSetup& es)
{ edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  OrderedHitPairs pairs;
  pairs.reserve(30000);
  thePairGenerator->hitPairs(region,pairs,ev,es);
  if (pairs.empty())
    return;
  
  int size = theLayers.size();

  std::vector<KDTreeNodeInfo<RecHitsSortedInPhi::HitIter> > layerTree; // re-used throughout
  std::vector<KDTreeLinkerAlgo<RecHitsSortedInPhi::HitIter> > hitTree(size);
  std::vector<float> rzError(size,0.0f); //save maximum errors
  double maxphi = Geom::twoPi(), minphi = -maxphi; //increase to cater for any range

  map<const DetLayer*, LayerRZPredictions> mapPred;
  const RecHitsSortedInPhi **thirdHitMap = new const RecHitsSortedInPhi*[size];
  for(int il = 0; il < size; il++) {
    thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
    const DetLayer *layer = theLayers[il].detLayer();
    LayerRZPredictions &predRZ = mapPred[layer];
    predRZ.line.initLayer(layer);
    predRZ.helix1.initLayer(layer);
    predRZ.helix2.initLayer(layer);
    predRZ.line.initTolerance(extraHitRZtolerance);
    predRZ.helix1.initTolerance(extraHitRZtolerance);
    predRZ.helix2.initTolerance(extraHitRZtolerance);
    predRZ.rzPositionFixup = MatchedHitRZCorrectionFromBending(layer,tTopo);
    
    RecHitsSortedInPhi::Range hitRange = thirdHitMap[il]->all(); // Get iterators
    layerTree.clear();
    double minz=999999.0, maxz= -999999.0; // Initialise to extreme values in case no hits
    float maxErr=0.0f;
    bool barrelLayer = (theLayers[il].detLayer()->location() == GeomDetEnumerators::barrel);
    if (hitRange.first != hitRange.second)
      { minz = barrelLayer? hitRange.first->hit()->globalPosition().z() : hitRange.first->hit()->globalPosition().perp();
	maxz = minz; //In case there's only one hit on the layer
	for (RecHitsSortedInPhi::HitIter hi=hitRange.first; hi != hitRange.second; ++hi)
	  { double angle = hi->phi();
	    double myz = barrelLayer? hi->hit()->globalPosition().z() : hi->hit()->globalPosition().perp();
	    //use (phi,r) for endcaps rather than (phi,z)
	    if (myz < minz) { minz = myz;} else { if (myz > maxz) {maxz = myz;}}
	    float myerr = barrelLayer? hi->hit()->errorGlobalZ(): hi->hit()->errorGlobalR();
	    if (myerr > maxErr) { maxErr = myerr;}
	    layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle, myz)); // save it
	    if (angle < 0)  // wrap all points in phi
	      { layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle+Geom::twoPi(), myz));}
	    else
	      { layerTree.push_back(KDTreeNodeInfo<RecHitsSortedInPhi::HitIter>(hi, angle-Geom::twoPi(), myz));}
	  }
      }
    KDTreeBox phiZ(minphi, maxphi, minz-0.01, maxz+0.01);  // declare our bounds
    //add fudge factors in case only one hit and also for floating-point inaccuracy
    hitTree[il].build(layerTree, phiZ); // make KDtree
    rzError[il] = maxErr; //save error
  }
  
  double curv = PixelRecoUtilities::curvature(1. / region.ptMin(), es);
  
  for (OrderedHitPairs::const_iterator ip = pairs.begin(); ip != pairs.end(); ++ip) {
    GlobalPoint gp1 = ip->inner()->globalPosition();
    GlobalPoint gp2 = ip->outer()->globalPosition();

    PixelRecoLineRZ line(gp1, gp2);
    PixelRecoPointRZ point2(gp2.perp(), gp2.z());
    ThirdHitPredictionFromCircle predictionRPhi(gp1, gp2, extraHitRPhitolerance);

    Range generalCurvature = predictionRPhi.curvature(region.originRBound());
    if (!intersect(generalCurvature, Range(-curv, curv)))
      continue;

    for(int il = 0; il < size; il++) {
      if (hitTree[il].empty()) continue; // Don't bother if no hits
      const DetLayer *layer = theLayers[il].detLayer();
      bool barrelLayer = layer->location() == GeomDetEnumerators::barrel;
      
      Range curvature = generalCurvature;
      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2, useMScat);
      
      LayerRZPredictions &predRZ = mapPred.find(layer)->second;
      predRZ.line.initPropagator(&line);
      
      HelixRZ helix;
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
          Range z3s = predRZ.line.detRange();
          double z3 = z3s.first < 0 ? max(z3s.first, z3s.second)
	    : min(z3s.first, z3s.second);
          double maxCurvature = HelixRZ::maxCurvature(&predictionRPhi,
                                                      gp1.z(), gp2.z(), z3);
          if (!intersect(curvature, Range(-maxCurvature, maxCurvature)))
            continue;
        }
	
        helix = HelixRZ(&predictionRPhi, gp1.z(), gp2.z(), curvature.first);
        HelixRZ helix2(&predictionRPhi, gp1.z(), gp2.z(), curvature.second);
	
        predRZ.helix1.initPropagator(&helix);
        predRZ.helix2.initPropagator(&helix2);
	
        Range rzRanges[2] = { predRZ.helix1(), predRZ.helix2() };
        rzRange.first = min(rzRanges[0].first, rzRanges[1].first);
        rzRange.second = max(rzRanges[0].second, rzRanges[1].second);
	
        // if the allowed curvatures include a straight line,
        // this can give us another extremum for allowed r/z
        if (curvature.first * curvature.second < 0.0) {
          Range rzLineRange = predRZ.line();
          rzRange.first = min(rzRange.first, rzLineRange.first);
          rzRange.second = max(rzRange.second, rzLineRange.second);
        }
      } else {
        rzRange = predRZ.line();
      }

      if (rzRange.first >= rzRange.second)
        continue;

      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) { 
        float phi0 = ip->outer()->globalPosition().phi();
        phiRange = Range(phi0 - dphi, phi0 + dphi);
      } else {
        Range radius;
	
        if (barrelLayer) {
          radius = predRZ.line.detRange();
          if (!intersect(rzRange, predRZ.line.detSize()))
            continue;
        } else {
          radius = rzRange;
          if (!intersect(radius, predRZ.line.detSize()))
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
      
      typedef RecHitsSortedInPhi::Hit Hit;
      layerTree.clear(); // Now recover hits in bounding box...
      float prmin=phiRange.min(), prmax=phiRange.max(); //get contiguous range
      if ((prmax-prmin) > Geom::twoPi())
	{ prmax=Geom::pi(); prmin = -Geom::pi();}
      else
	{ while (prmax>maxphi) { prmin -= Geom::twoPi(); prmax -= Geom::twoPi();}
	  while (prmin<minphi) { prmin += Geom::twoPi(); prmax += Geom::twoPi();}
	  // This needs range -twoPi to +twoPi to work
	}
      if (barrelLayer) {
	Range regMax = predRZ.line.detRange();
	Range regMin = predRZ.line(regMax.min());
	regMax = predRZ.line(regMax.max());
	correction.correctRZRange(regMin);
	correction.correctRZRange(regMax);
	if (regMax.min() < regMin.min()) { swap(regMax, regMin);}
	KDTreeBox phiZ(prmin, prmax,
		       regMin.min()-fnSigmaRZ*rzError[il],
		       regMax.max()+fnSigmaRZ*rzError[il]);
	hitTree[il].search(phiZ, layerTree);
      }
      else {
	KDTreeBox phiZ(prmin, prmax,
		       rzRange.min()-fnSigmaRZ*rzError[il],
		       rzRange.max()+fnSigmaRZ*rzError[il]);
	hitTree[il].search(phiZ, layerTree);
      }
      
      MatchedHitRZCorrectionFromBending l2rzFixup(ip->outer()->det()->geographicalId(), tTopo);
      MatchedHitRZCorrectionFromBending l3rzFixup = predRZ.rzPositionFixup;
      for (std::vector<KDTreeNodeInfo<RecHitsSortedInPhi::HitIter> >::iterator ih = layerTree.begin();
	   ih !=layerTree.end(); ++ih) {
	
	const RecHitsSortedInPhi::HitIter KDdata = (*ih).data;
	GlobalPoint p3 = KDdata->hit()->globalPosition();
	double p3_r = p3.perp();
	double p3_z = p3.z();
	double p3_phi = p3.phi();

	Range rangeRPhi = predictionRPhi(curvature, p3_r);
	correction.correctRPhiRange(rangeRPhi);

	double phiErr = nSigmaPhi * sqrt(KDdata->hit()->globalPositionError().phierr(p3));
	if (!checkPhiInRange(p3_phi, rangeRPhi.first/p3_r - phiErr, rangeRPhi.second/p3_r + phiErr))
	  continue;
	
	const TransientTrackingRecHit::ConstRecHitPointer& hit = KDdata->hit();
	Basic2DVector<double> thc(p3.x(), p3.y());
	
	double curv_ = predictionRPhi.curvature(thc);
	double p2_r = point2.r(), p2_z = point2.z();

	l2rzFixup(predictionRPhi, curv_, *ip->outer(), p2_r, p2_z, tTopo);
	l3rzFixup(predictionRPhi, curv_, *hit, p3_r, p3_z, tTopo);

	Range rangeRZ;
	if (useBend) {
	  HelixRZ updatedHelix(&predictionRPhi, gp1.z(), p2_z, curv_);
	  rangeRZ = predRZ.helix1(barrelLayer ? p3_r : p3_z, updatedHelix);
	} else {
	  float tIP = predictionRPhi.transverseIP(thc);
	  PixelRecoPointRZ updatedPoint2(p2_r, p2_z);
	  PixelRecoLineRZ updatedLine(line.origin(), point2, tIP);
	  rangeRZ = predRZ.line(barrelLayer ? p3_r : p3_z, line);
	}
	correction.correctRZRange(rangeRZ);
	
	double err = nSigmaRZ * sqrt(barrelLayer
				     ? hit->globalPositionError().czz()
				     : hit->globalPositionError().rerr(p3));
	rangeRZ.first -= err, rangeRZ.second += err;

	if (!rangeRZ.inside(barrelLayer ? p3_z : p3_r))
	  continue;
	if (theMaxElement!=0 && result.size() >= theMaxElement) {
	  result.clear();
	  edm::LogError("TooManyTriplets")<<" number of triples exceed maximum. no triplets produced.";
	  delete[] thirdHitMap;
	  return;
	}
	result.push_back(OrderedHitTriplet(ip->inner(), ip->outer(), hit)); 
      }
    }
  }
  delete[] thirdHitMap;
}

bool PixelTripletLargeTipGenerator::checkPhiInRange(float phi, float phi1, float phi2) const
{ while (phi > phi2) phi -= 2. * M_PI;
  while (phi < phi1) phi += 2. * M_PI;
  return phi <= phi2;
}  

std::pair<float, float>
PixelTripletLargeTipGenerator::mergePhiRanges(const std::pair<float, float> &r1,
					      const std::pair<float, float> &r2) const
{ float r2Min = r2.first;
  float r2Max = r2.second;
  while (r1.first - r2Min > +M_PI) r2Min += 2. * M_PI, r2Max += 2. * M_PI;
  while (r1.first - r2Min < -M_PI) r2Min -= 2. * M_PI, r2Max -= 2. * M_PI;
  return std::make_pair(min(r1.first, r2Min), max(r1.second, r2Max));
}
