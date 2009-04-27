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
#include "ThirdHitZPrediction.h"

#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoPixelVertexing/PixelTriplets/src/PixelTripletHLTGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromInvParabola.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitRZPrediction.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "UserCode/konec/interface/Analysis.h"
#include "TProfile.h"
#include "TH1D.h"


#include <iostream>
using pixelrecoutilities::LongitudinalBendingCorrection;
typedef PixelRecoRange<float> Range;
template<class T> T sqr(T t) { return t * t;}

using namespace std;
using namespace ctfseeding;

PixelTripletNoTipGenerator:: PixelTripletNoTipGenerator(const edm::ParameterSet& cfg)
    : thePairGenerator(0),
      theLayerCache(0),
      extraHitRZtolerance(cfg.getParameter<double>("extraHitRZtolerance")),
      extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")),
      theNSigma(cfg.getParameter<double>("nSigma")),
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

  edm::Handle<reco::BeamSpot> bsHandle;
  ev.getByLabel( theBeamSpotTag, bsHandle);
  if(!bsHandle.isValid()) return;
  const reco::BeamSpot & bs = *bsHandle;
  double errorXY = sqrt( sqr(bs.BeamWidthX()) + sqr(bs.BeamWidthY()) );

  GlobalPoint bsPoint = region.origin();
  GlobalVector shift =   bsPoint - GlobalPoint(0.,0.,0.);

  OrderedHitPairs pairs; pairs.reserve(30000);
  OrderedHitPairs::const_iterator ip;
  thePairGenerator->hitPairs(region,pairs,ev,es);

  if (pairs.size() ==0) return;

  int size = theLayers.size();

  const RecHitsSortedInPhi **thirdHitMap = new const RecHitsSortedInPhi*[size];
  for (int il=0; il <=size-1; il++) {
     thirdHitMap[il] = &(*theLayerCache)(&theLayers[il], region, ev, es);
  }

  const HitPairGeneratorFromLayerPair * pairGen = dynamic_cast<const HitPairGeneratorFromLayerPair *>(thePairGenerator);
  const DetLayer * firstLayer = pairGen->innerLayer().detLayer();
  const DetLayer * secondLayer = pairGen->outerLayer().detLayer();
  if (!firstLayer || !secondLayer) return;

  MultipleScatteringParametrisation sigma1RPhi( firstLayer, es);
  MultipleScatteringParametrisation sigma2RPhi( secondLayer, es);

  typedef RecHitsSortedInPhi::Hit Hit;
  for (ip = pairs.begin(); ip != pairs.end(); ip++) {

    GlobalPoint p1((*ip).inner()->globalPosition()-shift);
    GlobalPoint p2((*ip).outer()->globalPosition()-shift);

    ThirdHitPredictionFromInvLine  predictionRPhiTMP(p1, p2 );
    double pt_p1p2 = 1./PixelRecoUtilities::inversePt(predictionRPhiTMP.curvature(),es);

    PixelRecoPointRZ point1(p1.perp(), p1.z());
    PixelRecoPointRZ point2(p2.perp(), p2.z());

    PixelRecoLineRZ  line(point1, point2);
    double msRPhi1 = sigma1RPhi(pt_p1p2, line.cotLine());
    double msRPhi2 = sigma2RPhi(pt_p1p2,  line.cotLine(),point1);
    double sinTheta = 1/sqrt(1+sqr(line.cotLine()));
    double cosTheta = fabs(line.cotLine())/sqrt(1+sqr(line.cotLine()));

    double p1_errorRPhi = sqrt(sqr((*ip).inner()->errorGlobalRPhi())+sqr(msRPhi1) +sqr(errorXY));
    double p2_errorRPhi = sqrt(sqr((*ip).outer()->errorGlobalRPhi())+sqr(msRPhi2) +sqr(errorXY));

    ThirdHitPredictionFromInvLine  predictionRPhi(p1, p2, p1_errorRPhi, p2_errorRPhi );

    for (int il=0; il <=size-1; il++) {

      const DetLayer * layer = theLayers[il].detLayer();
      bool barrelLayer = (layer->location() == GeomDetEnumerators::barrel);
      MultipleScatteringParametrisation sigma3RPhi( layer, es);
      double msRPhi3 = sigma3RPhi(pt_p1p2, line.cotLine(),point2);

      Range rRange;
      if (barrelLayer) {
         const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(*layer);
         float halfThickness  = bl.surface().bounds().thickness()/2;
         float radius = bl.specificSurface().radius();
         rRange = Range(radius-halfThickness, radius+halfThickness);
      } else {
        const ForwardDetLayer& fl = dynamic_cast<const ForwardDetLayer&>(*layer);
        float halfThickness  = fl.surface().bounds().thickness()/2;
        float zLayer = fl.position().z() ;
        float zMin = zLayer-halfThickness;
        float zMax = zLayer+halfThickness;
        GlobalVector dr = p2-p1;
        GlobalPoint p3_a = p2+dr*(zMin-p2.z())/dr.z();
        GlobalPoint p3_b = p2+dr*(zMax-p2.z())/dr.z();
        if (zLayer * p3_a.z() < 0) continue;
        rRange = Range(p3_a.perp(), p3_b.perp());
        rRange.sort();
      }
      double displacment = shift.perp();
      GlobalPoint crossing1 = predictionRPhi.crossing(rRange.min()-displacment)+shift;
      GlobalPoint crossing2 = predictionRPhi.crossing(rRange.max()+displacment)+shift;
      float c1_phi= crossing1.phi(); 
      float c2_phi= crossing2.phi(); 
      if (c2_phi < c1_phi) swap(c1_phi,c2_phi); 
      if (c2_phi-c1_phi > M_PI) { c2_phi -= 2*M_PI;  swap(c1_phi,c2_phi); }
      double extraAngle = (displacment+5*msRPhi3)/rRange.min()+0.3;
      c1_phi -= extraAngle; 
      c2_phi += extraAngle;
//      std::cout <<"AFTER: phi1: "<< c1_phi<<" phi2: "<< c2_phi <<" MultScatt: "<<msRPhi3<<" fromBS: "<< sigma3RPhi(region.ptMin(), line.cotLine())<< endl;
//    vector<Hit> thirdHits = thirdHitMap[il]->hits();
      vector<Hit> thirdHits = thirdHitMap[il]->hits(c1_phi, c2_phi) ;

      typedef vector<Hit>::const_iterator IH;
      for (IH th=thirdHits.begin(), eh=thirdHits.end(); th < eh; ++th) {
        GlobalPoint p3((*th)->globalPosition()-shift);
        double p3_errorRPhi = sqrt(sqr((*th)->errorGlobalRPhi()) +sqr(msRPhi3) + sqr(errorXY));

        predictionRPhi.add(p3,p3_errorRPhi);

        double curvature = predictionRPhi.curvature();
           
        ThirdHitZPrediction zPrediction( (*ip).inner()->globalPosition(), sqrt(sqr((*ip).inner()->errorGlobalR())+sqr(msRPhi1/cosTheta)), sqrt( sqr((*ip).inner()->errorGlobalZ())+ sqr(msRPhi1/sinTheta)), 
                                         (*ip).outer()->globalPosition(), sqrt(sqr((*ip).outer()->errorGlobalR())+sqr(msRPhi2/cosTheta)), sqrt( sqr((*ip).outer()->errorGlobalZ())+sqr(msRPhi2/sinTheta)), 
                                          1./curvature, theNSigma);
         ThirdHitZPrediction::Range zRange = zPrediction((*th)->globalPosition(), sqrt(sqr((*th)->errorGlobalR()))+sqr(msRPhi3/cosTheta));
         
         double z3Hit = (*th)->globalPosition().z(); 
         double z3HitError = theNSigma*(sqrt(sqr((*th)->errorGlobalZ()) + sqr(msRPhi3/sinTheta) ))+extraHitRZtolerance; 
         Range hitZRange(z3Hit-z3HitError, z3Hit+z3HitError); 
         bool inside = hitZRange.hasIntersection(zRange); 

         double curvatureMS = PixelRecoUtilities::curvature(1./region.ptMin(),es);
         bool ptCut = (predictionRPhi.curvature()-3.*predictionRPhi.errorCurvature() < curvatureMS); 
         bool chi2Cut = (predictionRPhi.chi2() < 25.);
         if (inside 
             && ptCut
             && chi2Cut
             ) {
           result.push_back( OrderedHitTriplet( (*ip).inner(), (*ip).outer(), *th));
/*
           double pullCurv = (predictionRPhi.curvature()-curvatureMS)/predictionRPhi.errorCurvature();
           std::cout <<" curvatureMC: "<<curvatureMS<<" curvature: "<<predictionRPhi.curvature()<<" error: "<<predictionRPhi.errorCurvature()<<endl;
           
           static_cast<TH1*>(gHistos.FindObject("h_chi2TMP"))->Fill(chi2);
           double ptX = 1./PixelRecoUtilities::inversePt(curvature,es);
           static_cast<TH1*>(gHistos.FindObject("h_PtTMP"))->Fill( (ptMC-ptX)/ptMC);
           static_cast<TH1*>(gHistos.FindObject("h_PullCurv"))->Fill(pullCurv);
*/
         }
         predictionRPhi.remove(p3,p3_errorRPhi);
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
