#ifndef RecoPixelVertexing_PixelTriplets_ThirdHitCorrection_H
#define RecoPixelVertexing_PixelTriplets_ThirdHitCorrection_H

namespace edm {class EventSetup; }
namespace pixelrecoutilities { class LongitudinalBendingCorrection; }
class DetLayer;

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

class ThirdHitCorrection {
public:

  typedef PixelRecoRange<float> Range; 

  ThirdHitCorrection( 
      const edm::EventSetup &es, 
      float pt, 
      const DetLayer * layer,
      const PixelRecoLineRZ & line,
      const PixelRecoPointRZ & constraint,
      bool useMultipleScattering,
      bool useBendingCorrection);
 
  void correctRPhiRange( Range & range) const;
  void correctRZRange( Range & range) const;

private:
  bool theBarrel;
  float theCosTheta, theSinTheta;
  PixelRecoLineRZ theLine;
  float theMultScattCorrRPhi;
  bool theUseMultipleScattering;
  bool theUseBendingCorrection;
  pixelrecoutilities::LongitudinalBendingCorrection * theBendingCorrection;
  
};

#endif
