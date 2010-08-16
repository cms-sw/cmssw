#include "RecoPixelVertexing/PixelTriplets/src/ThirdHitCorrection.h"

#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

using namespace pixelrecoutilities;
template <class T> inline T sqr( T t) {return t*t;}


//namespace pixelrecoutilities { class LongitudinalBendingCorrection; }
//class MultipleScatteringParametrisation;
//  const MultipleScatteringParametrisation * theScattering;
//  const pixelrecoutilities::LongitudinalBendingCorrection * theBenging;

using pixelrecoutilities::LongitudinalBendingCorrection;

void ThirdHitCorrection::init(const edm::EventSetup& es, 
			 float pt,
			 const DetLayer * layer,
			 const PixelRecoLineRZ & line,
			 const PixelRecoPointRZ & constraint,
			 bool useMultipleScattering,
			 bool useBendingCorrection) {

  theUseMultipleScattering = useMultipleScattering;
  theUseBendingCorrection = useBendingCorrection;
  theLine = line;
  theMultScattCorrRPhi =0;
  theMScoeff=0;

  if (!theUseMultipleScattering && !theUseBendingCorrection) return;

  float overSinTheta = std::sqrt(1.f+sqr(line.cotLine()));
  float overCosTheta = overSinTheta/std::abs(line.cotLine());
  theBarrel = (layer->location() == GeomDetEnumerators::barrel);

  if (theUseMultipleScattering) {
    MultipleScatteringParametrisation sigmaRPhi(layer, es);
    theMultScattCorrRPhi = 3.f*sigmaRPhi(pt, line.cotLine(), constraint);
    if (theBarrel) {
      theMScoeff =  theMultScattCorrRPhi*overSinTheta; 
    } else {
      theMScoeff =  theMultScattCorrRPhi*overCosTheta;
    }
  }

  if (useBendingCorrection)  theBendingCorrection.init(pt,es);

}


void ThirdHitCorrection::correctRPhiRange( Range & range) const
{
  if (theUseMultipleScattering) {
    range.first -= theMultScattCorrRPhi;
    range.second += theMultScattCorrRPhi;
  }
}
void ThirdHitCorrection::correctRZRange( Range & range) const
{ 
  if (theUseMultipleScattering) {
    range.first -= theMScoeff;
    range.second += theMScoeff;
  } 

  if (theUseBendingCorrection) {
    if (theBarrel) {
      float cotTheta = theLine.cotLine();
      if (cotTheta > 0) {
        float radius = theLine.rAtZ(range.max());
        float corr = theBendingCorrection(radius) * cotTheta;
        range.second +=  corr;
      } else {
        float radius = theLine.rAtZ(range.min());
        float corr = theBendingCorrection(radius) * std::abs(cotTheta);
        range.first -=  corr;
      }
    } 
    else {
      float radius = range.max();
      float corr = theBendingCorrection(radius);
      range.first -= corr;
    }
  }
}
