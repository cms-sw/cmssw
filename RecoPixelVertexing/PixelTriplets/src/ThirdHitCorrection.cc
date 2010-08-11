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

ThirdHitCorrection::ThirdHitCorrection(const edm::EventSetup& es, 
      float pt,
      const DetLayer * layer,
      const PixelRecoLineRZ & line,
      const PixelRecoPointRZ & constraint,
      bool useMultipleScattering,
      bool useBendingCorrection)
  :
    theBarrel(false),
    theUseMultipleScattering(useMultipleScattering),
    theUseBendingCorrection( useBendingCorrection),
    theCosTheta(0.), theSinTheta(0.),
    theLine(line),
    theMultScattCorrRPhi(0.),
    theBendingCorrection(pt,es){

  if (!theUseMultipleScattering && !theUseBendingCorrection) return;
  theSinTheta = 1.f/std::sqrt(1.f+sqr(line.cotLine()));
  theCosTheta = std::abs(line.cotLine())*theSinTheta;
  theBarrel = (layer->location() == GeomDetEnumerators::barrel);

  if (theUseMultipleScattering) {
    MultipleScatteringParametrisation sigmaRPhi(layer, es);
    theMultScattCorrRPhi = 3.f*sigmaRPhi(pt, line.cotLine(), constraint); 
  }

}

ThirdHitCorrection::~ThirdHitCorrection(){}

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
    float extra = 0.f;
    if (theBarrel) {
      if (theSinTheta > 1.e-5f) extra =  theMultScattCorrRPhi/theSinTheta; 
    } else {
      if (theCosTheta > 1.e-5f) extra =  theMultScattCorrRPhi/theCosTheta;
    }
    range.first -= extra;
    range.second += extra;
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
