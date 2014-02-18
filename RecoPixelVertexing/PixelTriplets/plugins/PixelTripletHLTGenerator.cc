#include "RecoPixelVertexing/PixelTriplets/plugins/PixelTripletHLTGenerator.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"

#include "ThirdHitPredictionFromInvParabola.h"
#include "ThirdHitRZPrediction.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "ThirdHitCorrection.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
//#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"
//#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerAlgo.h" //amend to point at your copy...
#include "RecoPixelVertexing/PixelTriplets/plugins/KDTreeLinkerTools.h"

#include<cstdio>

using pixelrecoutilities::LongitudinalBendingCorrection;
typedef PixelRecoRange<float> Range;

using namespace std;
using namespace ctfseeding;

PixelTripletHLTGenerator:: PixelTripletHLTGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
  : HitTripletGeneratorFromPairAndLayers(cfg),
    useFixedPreFiltering(cfg.getParameter<bool>("useFixedPreFiltering")),
    extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
    extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
    useMScat(cfg.getParameter<bool>("useMultScattering")),
    useBend(cfg.getParameter<bool>("useBending"))
{
  dphi =  (useFixedPreFiltering) ?  cfg.getParameter<double>("phiPreFiltering") : 0;
  
  edm::ParameterSet comparitorPSet =
    cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  if(comparitorName != "none") {
    theComparitor.reset( SeedComparitorFactory::get()->create( comparitorName, comparitorPSet, iC) );
  }
}

PixelTripletHLTGenerator::~PixelTripletHLTGenerator() {}

void PixelTripletHLTGenerator::hitTriplets(const TrackingRegion& region, 
					   OrderedHitTriplets & result,
					   const edm::Event & ev,
					   const edm::EventSetup& es,
					   SeedingLayerSetsHits::SeedingLayerSet pairLayers,
					   const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers)
{

  if (theComparitor) theComparitor->init(ev, es);
  
  auto const & doublets = thePairGenerator->doublets(region,ev,es, pairLayers);
  
  if (doublets.empty()) return;

  auto outSeq =  doublets.detLayer(HitDoublets::outer)->seqNum();


  // std::cout << "pairs " << doublets.size() << std::endl;
  
  float regOffset = region.origin().perp(); //try to take account of non-centrality (?)
  int size = thirdLayers.size();
  
  #ifdef __clang__
  std::vector<ThirdHitRZPrediction<PixelRecoLineRZ>> preds(size);
  #else
  ThirdHitRZPrediction<PixelRecoLineRZ> preds[size];
  #endif
  
  const RecHitsSortedInPhi * thirdHitMap[size];
  typedef RecHitsSortedInPhi::Hit Hit;

  using NodeInfo = KDTreeNodeInfo<unsigned int>;
  std::vector<NodeInfo > layerTree; // re-used throughout
  std::vector<unsigned int> foundNodes; // re-used thoughout
  foundNodes.reserve(100);

  #ifdef __clang__
  std::vector<KDTreeLinkerAlgo<unsigned int>> hitTree(size);
  #else
  KDTreeLinkerAlgo<unsigned int> hitTree[size];
  #endif
  float rzError[size]; //save maximum errors
  float maxphi = Geom::ftwoPi(), minphi = -maxphi; // increase to cater for any range
  
  // fill the prediction vector
  for (int il=0; il!=size; ++il) {
    thirdHitMap[il] = &(*theLayerCache)(thirdLayers[il], region, ev, es);
    auto const & hits = *thirdHitMap[il];
    ThirdHitRZPrediction<PixelRecoLineRZ> & pred = preds[il];
    pred.initLayer(thirdLayers[il].detLayer());
    pred.initTolerance(extraHitRZtolerance);
    
    layerTree.clear();
    float minv=999999.0, maxv= -999999.0; // Initialise to extreme values in case no hits
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
    // std::cout << "layer " << thirdLayers[il].detLayer()->seqNum() << " " << layerTree.size() << std::endl; 
  }
  
  float imppar = region.originRBound();
  float imppartmp = region.originRBound()+region.origin().perp();
  float curv = PixelRecoUtilities::curvature(1.f/region.ptMin(), es);
  
  for (std::size_t ip =0;  ip!=doublets.size(); ip++) {
    auto xi = doublets.x(ip,HitDoublets::inner);
    auto yi = doublets.y(ip,HitDoublets::inner);
    auto zi = doublets.z(ip,HitDoublets::inner);
    auto rvi = doublets.rv(ip,HitDoublets::inner);
    auto xo = doublets.x(ip,HitDoublets::outer);
    auto yo = doublets.y(ip,HitDoublets::outer);
    auto zo = doublets.z(ip,HitDoublets::outer);
    auto rvo = doublets.rv(ip,HitDoublets::outer);
    
    PixelRecoPointRZ point1(rvi, zi);
    PixelRecoPointRZ point2(rvo, zo);
    PixelRecoLineRZ  line(point1, point2);
    ThirdHitPredictionFromInvParabola predictionRPhi(xi-region.origin().x(),yi-region.origin().y(),
						     xo-region.origin().x(),yo-region.origin().y(),
						     imppar,curv,extraHitRPhitolerance);

    ThirdHitPredictionFromInvParabola predictionRPhitmp(xi,yi,xo,yo,imppartmp,curv,extraHitRPhitolerance);

    // printf("++Constr %f %f %f %f %f %f %f\n",xi,yi,xo,yo,imppartmp,curv,extraHitRPhitolerance);     

    // std::cout << ip << ": " << point1.r() << ","<< point1.z() << " " 
    //                        << point2.r() << ","<< point2.z() <<std::endl;

    for (int il=0; il!=size; ++il) {
      if (hitTree[il].empty()) continue; // Don't bother if no hits
      
      auto const & hits = *thirdHitMap[il];
      
      const DetLayer * layer = thirdLayers[il].detLayer();
      auto barrelLayer = layer->isBarrel();

      ThirdHitCorrection correction(es, region.ptMin(), layer, line, point2, outSeq, useMScat, useBend); 
      
      ThirdHitRZPrediction<PixelRecoLineRZ> & predictionRZ =  preds[il];
      
      predictionRZ.initPropagator(&line);
      Range rzRange = predictionRZ();
      correction.correctRZRange(rzRange);
      
      Range phiRange;
      if (useFixedPreFiltering) { 
	float phi0 = doublets.phi(ip,HitDoublets::outer);
	phiRange = Range(phi0-dphi,phi0+dphi);
      }
      else {
	Range radius;
	if (barrelLayer) {
	  radius =  predictionRZ.detRange();
	} else {
	  radius = Range(max(rzRange.min(), predictionRZ.detSize().min()),
			 min(rzRange.max(), predictionRZ.detSize().max()) );
	}
	if (radius.empty()) continue;

	// std::cout << "++R " << radius.min() << " " << radius.max()  << std::endl;

	/*
	Range rPhi1m = predictionRPhitmp(radius.max(), -1);
	Range rPhi1p = predictionRPhitmp(radius.max(),  1);
	Range rPhi2m = predictionRPhitmp(radius.min(), -1);
	Range rPhi2p = predictionRPhitmp(radius.min(),  1);
	Range rPhi1 = rPhi1m.sum(rPhi1p);
	Range rPhi2 = rPhi2m.sum(rPhi2p);
	
	
	auto rPhi1N = predictionRPhitmp(radius.max());
	auto rPhi2N = predictionRPhitmp(radius.min());

	std::cout << "VI " 
		  << rPhi1N.first <<'/'<< rPhi1.first << ' '
		  << rPhi1N.second <<'/'<< rPhi1.second << ' '
		  << rPhi2N.first <<'/'<< rPhi2.first << ' '
		  << rPhi2N.second <<'/'<< rPhi2.second
		  << std::endl;
	
	*/

	auto rPhi1 = predictionRPhitmp(radius.max());
	auto rPhi2 = predictionRPhitmp(radius.min());


	correction.correctRPhiRange(rPhi1);
	correction.correctRPhiRange(rPhi2);
	rPhi1.first  /= radius.max();
	rPhi1.second /= radius.max();
	rPhi2.first  /= radius.min();
	rPhi2.second /= radius.min();
	phiRange = mergePhiRanges(rPhi1,rPhi2);
      }
      
      constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f); // ...and continue as before
      constexpr float nSigmaPhi = 3.f;
      
      foundNodes.clear(); // Now recover hits in bounding box...
      float prmin=phiRange.min(), prmax=phiRange.max();
      if ((prmax-prmin) > Geom::ftwoPi())
	{ prmax=Geom::fpi(); prmin = -Geom::fpi();}
      else
	{ while (prmax>maxphi) { prmin -= Geom::ftwoPi(); prmax -= Geom::ftwoPi();}
	  while (prmin<minphi) { prmin += Geom::ftwoPi(); prmax += Geom::ftwoPi();}
	  // This needs range -twoPi to +twoPi to work
	}
      if (barrelLayer)
	{
	  Range regMax = predictionRZ.detRange();
	  Range regMin = predictionRZ(regMax.min()-regOffset);
	  regMax = predictionRZ(regMax.max()+regOffset);
	  correction.correctRZRange(regMin);
	  correction.correctRZRange(regMax);
	  if (regMax.min() < regMin.min()) { swap(regMax, regMin);}
	  KDTreeBox phiZ(prmin, prmax, regMin.min()-nSigmaRZ*rzError[il], regMax.max()+nSigmaRZ*rzError[il]);
	  hitTree[il].search(phiZ, foundNodes);
	}
      else
	{
	  KDTreeBox phiZ(prmin, prmax,
			 rzRange.min()-regOffset-nSigmaRZ*rzError[il],
			 rzRange.max()+regOffset+nSigmaRZ*rzError[il]);
	  hitTree[il].search(phiZ, foundNodes);
	}

      // std::cout << ip << ": " << thirdLayers[il].detLayer()->seqNum() << " " << foundNodes.size() << " " << prmin << " " << prmax << std::endl;


      // int kk=0;
      for (auto KDdata : foundNodes) {
	
	if (theMaxElement!=0 && result.size() >= theMaxElement){
	  result.clear();
	  edm::LogError("TooManyTriplets")<<" number of triples exceeds maximum. no triplets produced.";
	  return;
	}
	
	float p3_u = hits.u[KDdata]; 
	float p3_v =  hits.v[KDdata]; 
	float p3_phi =  hits.lphi[KDdata]; 

       //if ((kk++)%100==0)
       //std::cout << kk << ": " << p3_u << " " << p3_v << " " << p3_phi << std::endl;

	
	Range allowed = predictionRZ(p3_u);
	correction.correctRZRange(allowed);
	float vErr = nSigmaRZ *hits.dv[KDdata];
	Range hitRange(p3_v-vErr, p3_v+vErr);
	Range crossingRange = allowed.intersection(hitRange);
	if (crossingRange.empty())  continue;
	
	float ir = 1.f/hits.rv(KDdata);
	float phiErr = nSigmaPhi * hits.drphi[KDdata]*ir;
	for (int icharge=-1; icharge <=1; icharge+=2) {
	  Range rangeRPhi = predictionRPhi(hits.rv(KDdata), icharge);
	  correction.correctRPhiRange(rangeRPhi);
	  if (checkPhiInRange(p3_phi, rangeRPhi.first*ir-phiErr, rangeRPhi.second*ir+phiErr)) {
	    // insert here check with comparitor
	    OrderedHitTriplet hittriplet( doublets.hit(ip,HitDoublets::inner), doublets.hit(ip,HitDoublets::outer), hits.theHits[KDdata].hit());
	    if (!theComparitor  || theComparitor->compatible(hittriplet,region) ) {
	      result.push_back( hittriplet );
	    } else {
	      LogDebug("RejectedTriplet") << "rejected triplet from comparitor ";
	    }
	    break;
	  } 
	}
      }
    }
  }
  // std::cout << "triplets " << result.size() << std::endl;
}

bool PixelTripletHLTGenerator::checkPhiInRange(float phi, float phi1, float phi2) const
{
  while (phi > phi2) phi -=  Geom::ftwoPi();
  while (phi < phi1) phi +=  Geom::ftwoPi();
  return (  (phi1 <= phi) && (phi <= phi2) );
}  

std::pair<float,float> PixelTripletHLTGenerator::mergePhiRanges(const std::pair<float,float>& r1,
								const std::pair<float,float>& r2) const 
{ float r2_min=r2.first;
  float r2_max=r2.second;
  while (r1.first-r2_min > Geom::fpi()) { r2_min += Geom::ftwoPi(); r2_max += Geom::ftwoPi();}
  while (r1.first-r2_min < -Geom::fpi()) { r2_min -= Geom::ftwoPi();  r2_max -= Geom::ftwoPi(); }
  
  return std::make_pair(min(r1.first,r2_min),max(r1.second,r2_max));
}
