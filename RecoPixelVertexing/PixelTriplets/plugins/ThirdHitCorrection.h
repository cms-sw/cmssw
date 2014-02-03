#ifndef RecoPixelVertexing_PixelTriplets_ThirdHitCorrection_H
#define RecoPixelVertexing_PixelTriplets_ThirdHitCorrection_H

namespace edm {class EventSetup; }
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"

class DetLayer;

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

class ThirdHitCorrection {
public:

  typedef PixelRecoRange<float> Range; 

  ThirdHitCorrection(){}

  ThirdHitCorrection( 
      const edm::EventSetup &es, 
      float pt, 
      const DetLayer * layer,
      const PixelRecoLineRZ & line,
      const PixelRecoPointRZ & constraint, int ol,
      bool useMultipleScattering,
      bool useBendingCorrection = false) 
  { 
    init(es, pt, layer, line, constraint, ol, useMultipleScattering, useBendingCorrection);
  }

  void init( 
	    const edm::EventSetup &es, 
      float pt, 
      const DetLayer * layer,
      const PixelRecoLineRZ & line,
      const PixelRecoPointRZ & constraint, int ol,
      bool useMultipleScattering,
      bool useBendingCorrection = false);


  ~ThirdHitCorrection(){}
 
  void correctRPhiRange( Range & range) const;
  void correctRZRange( Range & range) const;

private:
  bool theBarrel;

  bool theUseMultipleScattering;
  bool theUseBendingCorrection;

  PixelRecoLineRZ theLine;
  float theMultScattCorrRPhi;
  float theMScoeff;

  pixelrecoutilities::LongitudinalBendingCorrection theBendingCorrection;
  
};

#endif
