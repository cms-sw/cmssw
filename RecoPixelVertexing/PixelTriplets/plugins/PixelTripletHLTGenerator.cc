#include "RecoPixelVertexing/PixelTriplets/plugins/PixelTripletHLTGenerator.h"

#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
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
  
  edm::ParameterSet comparitorPSet =
    cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  theComparitor = (comparitorName == "none") ?
    0 :  SeedComparitorFactory::get()->create( comparitorName, comparitorPSet);
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

void PixelTripletHLTGenerator::hitTriplets(const TrackingRegion& region, 
					   OrderedHitTriplets & result,
					   const edm::Event & ev,
					   const edm::EventSetup& es)
{
  if (theComparitor) theComparitor->init(es);
  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  
  thePairGenerator->hitPairs(region,pairs,ev,es);
  
  if (pairs.empty()) return;
  
  float regOffset = region.origin().perp(); //try to take account of non-centrality (?)
  int size = theLayers.size();
  
  typedef std::vector<ThirdHitRZPrediction<PixelRecoLineRZ> >  Preds;
  Preds preds(size);
  
  std::vector<const RecHitsSortedInPhi *> thirdHitMap(size);
  typedef RecHitsSortedInPhi::Hit Hit;
 
  std::vector<KDTreeNodeInfo<RecHitsSortedInPhi::HitIter> > layerTree; // re-used throughout
  std::vector<KDTreeLinkerAlgo<RecHitsSortedInPhi::HitIter> > hitTree(size);
  std::vector<float> rzError(size,0.0f); //save maximum errors
  double maxphi = Geom::twoPi(), minphi = -maxphi; // increase to cater for any range
  
  // fill the prediction vector
  for (int il=0; il!=size; ++il) {
    thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
    ThirdHitRZPrediction<PixelRecoLineRZ> & pred = preds[il];
    pred.initLayer(theLayers[il].detLayer());
    pred.initTolerance(extraHitRZtolerance);
    
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
     if (hitTree[il].empty()) continue; // Don't bother if no hits
     const DetLayer * layer = theLayers[il].detLayer();
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
	 radius = Range(max(rzRange.min(), predictionRZ.detSize().min()),
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
     
     static float nSigmaRZ = std::sqrt(12.f); // ...and continue as before
     static float nSigmaPhi = 3.f;

     layerTree.clear(); // Now recover hits in bounding box...
     float prmin=phiRange.min(), prmax=phiRange.max();
     if ((prmax-prmin) > Geom::twoPi())
       { prmax=Geom::pi(); prmin = -Geom::pi();}
     else
       { while (prmax>maxphi) { prmin -= Geom::twoPi(); prmax -= Geom::twoPi();}
	 while (prmin<minphi) { prmin += Geom::twoPi(); prmax += Geom::twoPi();}
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
	 hitTree[il].search(phiZ, layerTree);
       }
     else
       {
	 KDTreeBox phiZ(prmin, prmax,
			rzRange.min()-regOffset-nSigmaRZ*rzError[il],
			rzRange.max()+regOffset+nSigmaRZ*rzError[il]);
	 hitTree[il].search(phiZ, layerTree);
       }
     for (std::vector<KDTreeNodeInfo<RecHitsSortedInPhi::HitIter> >::iterator ih = layerTree.begin();
	  ih !=layerTree.end(); ++ih) {
       
       if (theMaxElement!=0 && result.size() >= theMaxElement){
	 result.clear();
	 edm::LogError("TooManyTriplets")<<" number of triples exceeds maximum. no triplets produced.";
	 return;
       }
       
       const RecHitsSortedInPhi::HitIter KDdata = (*ih).data;
       GlobalPoint point(KDdata->hit()->globalPosition().x()-region.origin().x(),
			 KDdata->hit()->globalPosition().y()-region.origin().y(),
			 KDdata->hit()->globalPosition().z() ); 
       float p3_r = point.perp();
       float p3_z = point.z();
       float p3_phi = point.phi();
       if (barrelLayer) {
	 Range allowedZ = predictionRZ(p3_r);
	 correction.correctRZRange(allowedZ);
	 float zErr = nSigmaRZ * KDdata->hit()->errorGlobalZ();
	 Range hitRange(p3_z-zErr, p3_z+zErr);
	 Range crossingRange = allowedZ.intersection(hitRange);
	 if (crossingRange.empty())  continue;
       } else {
	 Range allowedR = predictionRZ(p3_z);
	 correction.correctRZRange(allowedR); 
	 float rErr = nSigmaRZ * KDdata->hit()->errorGlobalR();
	 Range hitRange(p3_r-rErr, p3_r+rErr);
	 Range crossingRange = allowedR.intersection(hitRange);
	 if ( crossingRange.empty())  continue;
       }
       
       float phiErr = nSigmaPhi * KDdata->hit()->errorGlobalRPhi()/p3_r;
       for (int icharge=-1; icharge <=1; icharge+=2) {
	 Range rangeRPhi = predictionRPhi(p3_r, icharge);
	 correction.correctRPhiRange(rangeRPhi);
	 if (checkPhiInRange(p3_phi, rangeRPhi.first/p3_r-phiErr, rangeRPhi.second/p3_r+phiErr)) {
	   // insert here check with comparitor
	   OrderedHitTriplet hittriplet( (*ip).inner(), (*ip).outer(), KDdata->hit());
	   if (!theComparitor  || theComparitor->compatible(hittriplet,region) ) {
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

std::pair<float,float> PixelTripletHLTGenerator::mergePhiRanges(const std::pair<float,float>& r1,
								const std::pair<float,float>& r2) const 
{ float r2_min=r2.first;
  float r2_max=r2.second;
  while (r1.first-r2_min > Geom::fpi()) { r2_min += Geom::ftwoPi(); r2_max += Geom::ftwoPi();}
  while (r1.first-r2_min < -Geom::fpi()) { r2_min -= Geom::ftwoPi();  r2_max -= Geom::ftwoPi(); }
  
  return std::make_pair(min(r1.first,r2_min),max(r1.second,r2_max));
}
