#include "RecoPixelVertexing/PixelTriplets/plugins/PixelTripletLargeTipGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "ThirdHitRZPrediction.h"
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
					 LayerCacheType *layerCache)
{
  thePairGenerator = pairs.clone();
  theLayerCache = layerCache;
}

void PixelTripletLargeTipGenerator::setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet pairLayers,
                                                     std::vector<SeedingLayerSetsHits::SeedingLayer> thirdLayers) {
  thePairGenerator->setSeedingLayers(pairLayers);
  theLayers = thirdLayers;
}

namespace {
  inline
  bool intersect(Range &range, const Range &second)
  {
    if (range.first > second.max() || range.second < second.min())
      return false;
    if (range.first < second.min())
      range.first = second.min();
    if (range.second > second.max())
      range.second = second.max();
    return range.first < range.second;
  }
}

void PixelTripletLargeTipGenerator::hitTriplets(const TrackingRegion& region, 
						OrderedHitTriplets & result,
						const edm::Event & ev,
						const edm::EventSetup& es)
{ 
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  auto const & doublets = thePairGenerator->doublets(region,ev,es);
  
  if (doublets.empty()) return;
   
  auto outSeq =  doublets.detLayer(HitDoublets::outer)->seqNum();


  int size = theLayers.size();


  using NodeInfo = KDTreeNodeInfo<unsigned int>;
  std::vector<NodeInfo > layerTree; // re-used throughout
  std::vector<unsigned int> foundNodes; // re-used throughout
  foundNodes.reserve(100);
  KDTreeLinkerAlgo<unsigned int> hitTree[size];

  float rzError[size]; //save maximum errors
  float maxphi = Geom::ftwoPi(), minphi = -maxphi; //increase to cater for any range

  LayerRZPredictions mapPred[size];

  const RecHitsSortedInPhi * thirdHitMap[size];

  for(int il = 0; il < size; il++) {
    thirdHitMap[il] = &(*theLayerCache)(theLayers[il], region, ev, es);
    auto const & hits = *thirdHitMap[il];
 
    const DetLayer *layer = theLayers[il].detLayer();
    LayerRZPredictions &predRZ = mapPred[il];
    predRZ.line.initLayer(layer);
    predRZ.helix1.initLayer(layer);
    predRZ.helix2.initLayer(layer);
    predRZ.line.initTolerance(extraHitRZtolerance);
    predRZ.helix1.initTolerance(extraHitRZtolerance);
    predRZ.helix2.initTolerance(extraHitRZtolerance);
    predRZ.rzPositionFixup = MatchedHitRZCorrectionFromBending(layer,tTopo);
    
    layerTree.clear();
    float minv=999999.0; float maxv = -999999.0; // Initialise to extreme values in case no hits
    float maxErr=0.0f;
    for (unsigned int i=0; i!=hits.size(); ++i) {
      auto angle = hits.phi(i);
      auto v =  hits.gv(i);
      //use (phi,r) for endcaps rather than (phi,z)
      minv = std::min(minv,v);  maxv = std::max(maxv,v);
      float myerr = hits.dv[i];
      maxErr = std::max(maxErr,myerr);
      layerTree.emplace_back(i, angle, v); // save it
      if (angle < 0)  // wrap all points in phi
	{ layerTree.emplace_back(i, angle+Geom::ftwoPi(), v);}
      else
	{ layerTree.emplace_back(i, angle-Geom::ftwoPi(), v);}
    }
    KDTreeBox phiZ(minphi, maxphi, minv-0.01f, maxv+0.01f);  // declare our bounds
    //add fudge factors in case only one hit and also for floating-point inaccuracy
    hitTree[il].build(layerTree, phiZ); // make KDtree
    rzError[il] = maxErr; //save error
  }

  double curv = PixelRecoUtilities::curvature(1. / region.ptMin(), es);
  
  for (std::size_t ip =0;  ip!=doublets.size(); ip++) {
    auto xi = doublets.x(ip,HitDoublets::inner);
    auto yi = doublets.y(ip,HitDoublets::inner);
    auto zi = doublets.z(ip,HitDoublets::inner);
    // auto rvi = doublets.rv(ip,HitDoublets::inner);
    auto xo = doublets.x(ip,HitDoublets::outer);
    auto yo = doublets.y(ip,HitDoublets::outer);
    auto zo = doublets.z(ip,HitDoublets::outer);
    // auto rvo = doublets.rv(ip,HitDoublets::outer);
    GlobalPoint gp1(xi,yi,zi);
    GlobalPoint gp2(xo,yo,zo);

    PixelRecoLineRZ line(gp1, gp2);
    PixelRecoPointRZ point2(gp2.perp(), zo);
    ThirdHitPredictionFromCircle predictionRPhi(gp1, gp2, extraHitRPhitolerance);

    Range generalCurvature = predictionRPhi.curvature(region.originRBound());
    if (!intersect(generalCurvature, Range(-curv, curv))) continue;

    for(int il = 0; il < size; il++) {
      if (hitTree[il].empty()) continue; // Don't bother if no hits
      const DetLayer *layer = theLayers[il].detLayer();
      bool barrelLayer = layer->isBarrel();
      
      Range curvature = generalCurvature;
      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2,  outSeq, useMScat);
      
      LayerRZPredictions &predRZ = mapPred[il];
      predRZ.line.initPropagator(&line);
      
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
          double z3 = z3s.first < 0 ? std::max(z3s.first, z3s.second)
	    : std::min(z3s.first, z3s.second);
          double maxCurvature = HelixRZ::maxCurvature(&predictionRPhi,
                                                      gp1.z(), gp2.z(), z3);
          if (!intersect(curvature, Range(-maxCurvature, maxCurvature)))
            continue;
        }
	
        HelixRZ helix1(&predictionRPhi, gp1.z(), gp2.z(), curvature.first);
        HelixRZ helix2(&predictionRPhi, gp1.z(), gp2.z(), curvature.second);
	
        predRZ.helix1.initPropagator(&helix1);
        predRZ.helix2.initPropagator(&helix2);
	
        Range rzRanges[2] = { predRZ.helix1(), predRZ.helix2() };
        predRZ.helix1.initPropagator(nullptr);
        predRZ.helix2.initPropagator(nullptr);

        rzRange.first = std::min(rzRanges[0].first, rzRanges[1].first);
        rzRange.second = std::max(rzRanges[0].second, rzRanges[1].second);
	
        // if the allowed curvatures include a straight line,
        // this can give us another extremum for allowed r/z
        if (curvature.first * curvature.second < 0.0) {
          Range rzLineRange = predRZ.line();
          rzRange.first = std::min(rzRange.first, rzLineRange.first);
          rzRange.second = std::max(rzRange.second, rzLineRange.second);
        }
      } else {
        rzRange = predRZ.line();
      }

      if (rzRange.first >= rzRange.second)
        continue;

      correction.correctRZRange(rzRange);

      Range phiRange;
      if (useFixedPreFiltering) {
	float phi0 = doublets.phi(ip,HitDoublets::outer);
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
      
      foundNodes.clear(); // Now recover hits in bounding box...
      float prmin=phiRange.min(), prmax=phiRange.max(); //get contiguous range
      if ((prmax-prmin) > Geom::ftwoPi())
	{ prmax=Geom::fpi(); prmin = -Geom::fpi();}
      else
	{ while (prmax>maxphi) { prmin -= Geom::ftwoPi(); prmax -= Geom::ftwoPi();}
	  while (prmin<minphi) { prmin += Geom::ftwoPi(); prmax += Geom::ftwoPi();}
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
	hitTree[il].search(phiZ, foundNodes);
      }
      else {
	KDTreeBox phiZ(prmin, prmax,
		       rzRange.min()-fnSigmaRZ*rzError[il],
		       rzRange.max()+fnSigmaRZ*rzError[il]);
	hitTree[il].search(phiZ, foundNodes);
      }
      
      MatchedHitRZCorrectionFromBending l2rzFixup(doublets.hit(ip,HitDoublets::outer)->det()->geographicalId(), tTopo);
      MatchedHitRZCorrectionFromBending l3rzFixup = predRZ.rzPositionFixup;

      thirdHitMap[il] = &(*theLayerCache)(theLayers[il], region, ev, es);
      auto const & hits = *thirdHitMap[il];
      for (auto KDdata : foundNodes) {
	GlobalPoint p3 = hits.gp(KDdata); 
	double p3_r = p3.perp();
	double p3_z = p3.z();
	float p3_phi =  hits.phi(KDdata); 

	Range rangeRPhi = predictionRPhi(curvature, p3_r);
	correction.correctRPhiRange(rangeRPhi);

	float ir = 1.f/p3_r;
	float phiErr = nSigmaPhi *  hits.drphi[KDdata]*ir;
	if (!checkPhiInRange(p3_phi, rangeRPhi.first*ir-phiErr, rangeRPhi.second*ir+phiErr))
	  continue;
	
	Basic2DVector<double> thc(p3.x(), p3.y());
	
	auto curv_ = predictionRPhi.curvature(thc);
	double p2_r = point2.r(); double p2_z = point2.z(); // they will be modified!
	
	l2rzFixup(predictionRPhi, curv_, *doublets.hit(ip,HitDoublets::outer), p2_r, p2_z, tTopo);
	l3rzFixup(predictionRPhi, curv_, *hits.theHits[KDdata].hit(), p3_r, p3_z, tTopo);
	
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
	
	double err = nSigmaRZ * hits.dv[KDdata];
	
	rangeRZ.first -= err, rangeRZ.second += err;
	
	if (!rangeRZ.inside(barrelLayer ? p3_z : p3_r)) continue;

	if (theMaxElement!=0 && result.size() >= theMaxElement) {
	  result.clear();
	  edm::LogError("TooManyTriplets")<<" number of triples exceed maximum. no triplets produced.";
	  return;
	}
	result.emplace_back( doublets.hit(ip,HitDoublets::inner), doublets.hit(ip,HitDoublets::outer), hits.theHits[KDdata].hit()); 
      }
    }
  }
  // std::cout << "found triplets " << result.size() << std::endl;
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
