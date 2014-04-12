#include "RecoTracker/ConversionSeedGenerators/interface/HitQuadrupletGeneratorFromLayerPairForPhotonConversion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace GeomDetEnumerators;
using namespace std;
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

//#define mydebug_QSeed

typedef PixelRecoRange<float> Range;
template<class T> inline T sqr( T t) {return t*t;}



#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Math/interface/deltaPhi.h"

HitQuadrupletGeneratorFromLayerPairForPhotonConversion::HitQuadrupletGeneratorFromLayerPairForPhotonConversion(
							     unsigned int inner,
							     unsigned int outer,
							     LayerCacheType* layerCache,
							     unsigned int nSize,
							     unsigned int max)
  : HitPairGenerator(nSize),
    theLayerCache(*layerCache), theOuterLayer(outer), theInnerLayer(inner)
{
  theMaxElement=max;
  ss = new std::stringstream;
}

void HitQuadrupletGeneratorFromLayerPairForPhotonConversion::hitPairs(const TrackingRegion & region, OrderedHitPairs & result,
								      const edm::Event& event, const edm::EventSetup& es)
{

  ss->str("");

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;
  typedef RecHitsSortedInPhi::Hit Hit;

  Layer innerLayerObj = innerLayer();
  Layer outerLayerObj = outerLayer();

  size_t totCountP2=0, totCountP1=0, totCountM2=0, totCountM1=0, selCount=0;
#ifdef mydebug_QSeed
  (*ss) << "In " << innerLayerObj.name() << " Out " << theOuterLayer.name() << std::endl;
#endif

  /*get hit sorted in phi for each layer: NB: doesn't apply any region cut*/
  const RecHitsSortedInPhi & innerHitsMap = theLayerCache(innerLayerObj, region, event, es);
  if (innerHitsMap.empty()) return;
 
  const RecHitsSortedInPhi& outerHitsMap = theLayerCache(outerLayerObj, region, event, es);
  if (outerHitsMap.empty()) return;
  /*----------------*/

  /*This object will check the compatibility of the his in phi among the two layers. */
  //InnerDeltaPhi deltaPhi(*innerLayerObj.detLayer(), region, es);

  vector<RecHitsSortedInPhi::Hit> innerHits;
  //  float outerPhimin, outerPhimax;
  float innerPhimin, innerPhimax;
  float maxDeltaPhi=1.; //sguazz
  //float maxDeltaPhi=1.;

  RecHitsSortedInPhi::Range outerHits = outerHitsMap.all();

  RecHitsSortedInPhi::HitIter nextoh;
  for (RecHitsSortedInPhi::HitIter oh = outerHits.first; oh!= outerHits.second; ++oh) { 
    RecHitsSortedInPhi::Hit ohit = (*oh).hit();
    GlobalPoint oPos = ohit->globalPosition();  

    totCountP2++;
    const HitRZCompatibility *checkRZ = region.checkRZ(outerLayerObj.detLayer(), ohit, es);
    for(nextoh=oh+1;nextoh!=outerHits.second; ++nextoh){
      
      RecHitsSortedInPhi::Hit nohit = (*nextoh).hit();
      GlobalPoint noPos = nohit->globalPosition();  
      
    
#ifdef mydebug_QSeed
      (*ss) << "\toPos " << oPos << " r " << oPos.perp() << " phi " << oPos.phi() <<  " cotTheta " << oPos.z()/oPos.perp() << std::endl;
      (*ss) << "\tnoPos " << noPos << " r " << noPos.perp() << " phi " << noPos.phi() <<  " cotTheta " << noPos.z()/noPos.perp() << std::endl;
      (*ss) << "\tDeltaPhi " << reco::deltaPhi(noPos.phi(),oPos.phi()) << std::endl;
#endif
    
    if(fabs(reco::deltaPhi(noPos.phi(),oPos.phi()))>maxDeltaPhi)  break;

    totCountM2++;
    
    /*Check the compatibility of the ohit with the eta of the seeding track*/
    if(failCheckRZCompatibility(nohit,*outerLayerObj.detLayer(),checkRZ,region)) continue;

    /*  
    //Do I need this? it uses a compatibility that probably I wouldn't 
    //Removing for the time being

    PixelRecoRange<float> phiRange = deltaPhi( oPos.perp(), oPos.phi(), oPos.z(), nSigmaPhi*(ohit->errorGlobalRPhi()));    
    if (phiRange.empty()) continue;
    */

  
    /*Get only the inner hits compatible with the conversion region*/
    innerPhimin=ohit->globalPosition().phi();
    innerPhimax=nohit->globalPosition().phi();
    // checkPhiRange(innerPhimin,innerPhimax);

    innerHits.clear();    
    innerHitsMap.hits(innerPhimin, innerPhimax, innerHits);

#ifdef mydebug_QSeed
    (*ss) << "\tiphimin, iphimax " << innerPhimin << " " << innerPhimax << std::endl;
#endif    

    const HitRZCompatibility *checkRZb = region.checkRZ(innerLayerObj.detLayer(),  ohit, es);
    const HitRZCompatibility *checkRZc = region.checkRZ(innerLayerObj.detLayer(), nohit, es);

    /*Loop on inner hits*/
    vector<RecHitsSortedInPhi::Hit>::const_iterator ieh = innerHits.end();
    for ( vector<RecHitsSortedInPhi::Hit>::const_iterator ih=innerHits.begin(); ih < ieh; ++ih) {  
      RecHitsSortedInPhi::Hit ihit = *ih;
      
#ifdef mydebug_QSeed
      GlobalPoint innPos = (*ih)->globalPosition();
      (*ss) << "\toPos " << oPos << " r " << oPos.perp() << " phi " << oPos.phi() <<  " cotTheta " << oPos.z()/oPos.perp() << std::endl;
      (*ss) << "\tnoPos " << noPos << " r " << noPos.perp() << " phi " << noPos.phi() <<  " cotTheta " << noPos.z()/noPos.perp() << std::endl;
      (*ss) << "\tinnPos " << innPos <<  " r " << innPos.perp() << " phi " << innPos.phi() << " cotTheta " << innPos.z()/innPos.perp() <<  std::endl;
#endif

      totCountP1++;
      
      /*Check the compatibility of the ihit with the two outer hits*/
      if(failCheckRZCompatibility(ihit,*innerLayerObj.detLayer(),checkRZb,region)
	 || 
	 failCheckRZCompatibility(ihit,*innerLayerObj.detLayer(),checkRZc,region) ) continue;
      
      
      for ( vector<RecHitsSortedInPhi::Hit>::const_iterator nextih=ih+1; nextih != ieh; ++nextih) {  
	RecHitsSortedInPhi::Hit nihit = *nextih;


#ifdef mydebug_QSeed
	GlobalPoint ninnPos = (*nextih)->globalPosition();
	(*ss) << "\toPos " << oPos << " r " << oPos.perp() << " phi " << oPos.phi() <<  " cotTheta " << oPos.z()/oPos.perp() << std::endl;
	(*ss) << "\tnoPos " << noPos << " r " << noPos.perp() << " phi " << noPos.phi() <<  " cotTheta " << noPos.z()/noPos.perp() << std::endl;
	(*ss) << "\tinnPos " << innPos <<  " r " << innPos.perp() << " phi " << innPos.phi() << " cotTheta " << innPos.z()/innPos.perp() <<  std::endl;
	(*ss) << "\tninnPos " << ninnPos <<  " r " << ninnPos.perp() << " phi " << ninnPos.phi() << " cotTheta " << ninnPos.z()/ninnPos.perp() <<  std::endl;
#endif

	totCountM1++;

	/*Check the compatibility of the nihit with the two outer hits*/
	if(failCheckRZCompatibility(nihit,*innerLayerObj.detLayer(),checkRZb,region)
	   || 
	   failCheckRZCompatibility(nihit,*innerLayerObj.detLayer(),checkRZc,region) ) continue;
	
	/*Sguazz modifica qui*/
	if(failCheckSlopeTest(ohit,nohit,ihit,nihit,region)) continue;

	if (theMaxElement!=0 && result.size() >= theMaxElement){
	  result.clear();
	  edm::LogError("TooManyQuads")<<"number of Quad combinations exceed maximum, no quads produced";
	  delete checkRZ;
	  delete checkRZb;
	  delete checkRZc;	  
#ifdef mydebug_QSeed
	  (*ss) << "In " << innerLayerObj.name() << " Out " << outerLayerObj.name()
		<< "\tP2 " << totCountP2 
		<< "\tM2 " << totCountM2 
		<< "\tP1 " << totCountP1 
		<< "\tM1 " << totCountM1 
		<< "\tsel " << selCount
		<< std::endl;
	  std::cout << (*ss).str();
#endif
	  return;
	}
	selCount++;
        result.push_back( OrderedHitPair( ihit, ohit) );
        result.push_back( OrderedHitPair( nihit, nohit) );
	//#ifdef mydebug_QSeed
	//(*ss) << "sizeOfresul " << result.size() << std::endl;
	//#endif
      }
    }
    delete checkRZb;
    delete checkRZc;
    }
    delete checkRZ;
  }
#ifdef mydebug_QSeed
  (*ss) << "In " << innerLayerObj.name() << " Out " << outerLayerObj.name()
	<< "\tP2 " << totCountP2 
	<< "\tM2 " << totCountM2 
	<< "\tP1 " << totCountP1 
	<< "\tM1 " << totCountM1 
	<< "\tsel " << selCount
	<< std::endl;
  std::cout << (*ss).str();
#endif
}




bool HitQuadrupletGeneratorFromLayerPairForPhotonConversion::
failCheckRZCompatibility(const RecHitsSortedInPhi::Hit& hit,const DetLayer& layer,const HitRZCompatibility *checkRZ,const TrackingRegion & region){

  if(!checkRZ) {
#ifdef mydebug_QSeed
    (*ss) << "*******\nNo valid checkRZ\n*******" << std::endl;
#endif
    return true;
  }

  static const float nSigmaRZ = std::sqrt(12.f);
  float r_reduced = std::sqrt( sqr(hit->globalPosition().x()-region.origin().x())+sqr(hit->globalPosition().y()-region.origin().y()));
  Range allowed;
  Range hitRZ;
  if (layer.location() == barrel) {
    allowed = checkRZ->range(r_reduced);
    float zErr = nSigmaRZ * hit->errorGlobalZ();
    hitRZ = Range(hit->globalPosition().z()-zErr, hit->globalPosition().z()+zErr);
  } else {
    allowed = checkRZ->range(hit->globalPosition().z());
    float rErr = nSigmaRZ * hit->errorGlobalR();
    hitRZ = Range(r_reduced-rErr, r_reduced+rErr);
  }
  Range crossRange = allowed.intersection(hitRZ);

#ifdef mydebug_QSeed      
  (*ss) 
    << "\n\t\t allowed Range " << allowed.min() << " \t, " << allowed.max() 
    << "\n\t\t hitRz   Range " << hitRZ.min()   << " \t, " << hitRZ.max() 
    << "\n\t\t Cross   Range " << crossRange.min()   << " \t, " << crossRange.max() 
    << std::endl;
  if( !crossRange.empty())
    (*ss) << "\n\t\t !!!!ACCEPTED!!! \n\n";
#endif
  
  return crossRange.empty();
}


bool HitQuadrupletGeneratorFromLayerPairForPhotonConversion::
failCheckSlopeTest(const RecHitsSortedInPhi::Hit & ohit, const RecHitsSortedInPhi::Hit & nohit, const RecHitsSortedInPhi::Hit & ihit, const RecHitsSortedInPhi::Hit & nihit, const TrackingRegion & region){

  double r[5], z[5], ez[5];
  //  double pr[2], pz[2], e2pz[2], mr[2], mz[2], e2mz[2];

  //
  //Hits
  r[0] = ohit->globalPosition().perp();
  z[0] = ohit->globalPosition().z();
  ez[0] = getEffectiveErrorOnZ(ohit, region);
  //
  r[1] = nohit->globalPosition().perp();
  z[1] = nohit->globalPosition().z();
  ez[1] = getEffectiveErrorOnZ(nohit, region);
  //
  r[2] = nihit->globalPosition().perp();
  z[2] = nihit->globalPosition().z();
  ez[2] = getEffectiveErrorOnZ(nihit, region);
  //
  r[3] = ihit->globalPosition().perp();
  z[3] = ihit->globalPosition().z();
  ez[3] = getEffectiveErrorOnZ(ihit, region);
  //
  //R (r) ordering of the 4 hit arrays
  bubbleSortVsR(4, r, z, ez);
  //
  //Vertex
  r[4] = region.origin().perp();
  z[4] = region.origin().z();
  double vError = region.originZBound();
  if ( vError > 15. ) vError = 1.;
  ez[4] = 3.*vError;

  //
  //Sequence of checks
  //
  //Inner segment == vertex
  double rInn = r[4];
  double zInnMin = z[4]-ez[4];
  double zInnMax = z[4]+ez[4];
  //
  // Int == 2, Out == 3
  double rOut = r[3];
  double zOutMin = z[3]-ez[3];
  double zOutMax = z[3]+ez[3];
  double rInt = r[2];
  double zIntMin = z[2]-ez[2];
  double zIntMax = z[2]+ez[2];
  if ( failCheckSegmentZCompatibility(rInn, zInnMin, zInnMax, 
				      rInt, zIntMin, zIntMax,
				      rOut, zOutMin, zOutMax) ) return true;
  //
  // Int == 1, Out == 2 (with updated limits)
  rOut = rInt;
  zOutMin = zIntMin;
  zOutMax = zIntMax;
  rInt    = r[1];
  zIntMin = z[1]-ez[1];
  zIntMax = z[1]+ez[1];
  if ( failCheckSegmentZCompatibility(rInn, zInnMin, zInnMax, 
				      rInt, zIntMin, zIntMax,
				      rOut, zOutMin, zOutMax) ) return true;
  //
  // Int == 0, Out == 1 (with updated limits)
  rOut = rInt;
  zOutMin = zIntMin;
  zOutMax = zIntMax;
  rInt    = r[0];
  zIntMin = z[0]-ez[0];
  zIntMax = z[0]+ez[0];
  if ( failCheckSegmentZCompatibility(rInn, zInnMin, zInnMax, 
				      rInt, zIntMin, zIntMax,
				      rOut, zOutMin, zOutMax) ) return true;

  //
  // Test is ok!!!
  return false;

}


double HitQuadrupletGeneratorFromLayerPairForPhotonConversion::verySimpleFit(int size, double* ax, double* ay, double* e2y, double& p0, double& e2p0, double& p1){
  
  //#include "RecoTracker/ConversionSeedGenerators/interface/verySimpleFit.icc"
  return 0;
}


double HitQuadrupletGeneratorFromLayerPairForPhotonConversion::getSqrEffectiveErrorOnZ(const RecHitsSortedInPhi::Hit & hit, const TrackingRegion & region){

  //
  //Fit-wise the effective error on Z is approximately the sum in quadrature of the error on Z 
  //and the error on R correctly projected by using hit-vertex direction

  double sqrProjFactor = sqr((hit->globalPosition().z()-region.origin().z())/(hit->globalPosition().perp()-region.origin().perp()));
  return (hit->globalPositionError().czz()+sqrProjFactor*hit->globalPositionError().rerr(hit->globalPosition()));

}

double HitQuadrupletGeneratorFromLayerPairForPhotonConversion::getEffectiveErrorOnZ(const RecHitsSortedInPhi::Hit & hit, const TrackingRegion & region){

  //
  //Fit-wise the effective error on Z is approximately the sum in quadrature of the error on Z 
  //and the error on R correctly projected by using hit-vertex direction
  double sqrProjFactor = sqr((hit->globalPosition().z()-region.origin().z())/(hit->globalPosition().perp()-region.origin().perp()));
  double effErr = sqrt(hit->globalPositionError().czz()+sqrProjFactor*hit->globalPositionError().rerr(hit->globalPosition())); 
  if ( effErr>2. ) {
    effErr*=1.8; //Single Side error is strip length * sqrt( 12.) = 3.46
    //empirically found that the error on ss hits value it is already twice as large (two sigmas)!
    //Multiply by sqrt(12.)/2.=1.73 to have effErr equal to the strip lenght (1.8 to allow for some margin)
    //effErr*=2.5; //Used in some tests
  } else {
    effErr*=2.; //Tight //Double side... allowing for 2 sigma variation 
    //effErr*=5.; //Loose //Double side... allowing for 2 sigma variation 
  }
  return effErr;

}

void HitQuadrupletGeneratorFromLayerPairForPhotonConversion::bubbleSortVsR(int n, double* ar, double* az, double* aez){
  bool swapped = true;
  int j = 0;
  double tmpr, tmpz, tmpez;
  while (swapped) {
    swapped = false;
    j++;
    for (int i = 0; i < n - j; i++) {
      if (  ar[i] > ar[i+1] ) {
	tmpr = ar[i];
	ar[i] = ar[i + 1];
	ar[i + 1] = tmpr;
	tmpz = az[i];
	az[i] = az[i + 1];
	az[i + 1] = tmpz;
	tmpez = aez[i];
	aez[i] = aez[i + 1];
	aez[i + 1] = tmpez;
	swapped = true;
      }
    }
  }
}

bool HitQuadrupletGeneratorFromLayerPairForPhotonConversion::
failCheckSegmentZCompatibility(double &rInn, double &zInnMin, double &zInnMax,
			       double &rInt, double &zIntMin, double &zIntMax,
			       double &rOut, double &zOutMin, double &zOutMax){
  //
  // Check the compatibility in z of an INTermediate segment between an INNer segment and an OUTer segment;
  // when true is returned zIntMin and zIntMax are replaced with allowed range values
  
  //Left side
  double zLeft = getZAtR(rInn, zInnMin, rInt, rOut, zOutMin);
  if ( zIntMax < zLeft ) return true; 
  //Right side
  double zRight = getZAtR(rInn, zInnMax, rInt, rOut, zOutMax);
  if ( zIntMin > zRight ) return true; 
  if ( zIntMin < zLeft && zIntMax < zRight ) {
    zIntMax = zLeft;
    return false;
  }
  if ( zIntMin > zLeft && zIntMax > zRight ) {
    zIntMax = zRight;
    return false;
  }
  
  //Segment is fully contained
  return false;

}

double HitQuadrupletGeneratorFromLayerPairForPhotonConversion::
getZAtR(double &rInn, double &zInn,
	double &r, 
	double &rOut, double &zOut){

  //    z - zInn      r - rInn                                  r - rInn 
  //  ----------- = ----------- ==> z = zInn + (zOut - zInn) * -----------
  //  zOut - zInn   rOut - rInn                                rOut - rInn

  
  return zInn + (zOut - zInn)*(r - rInn)/(rOut - rInn);

}
