#ifndef RecoPixelVertexing_PixelTriplets_ThirdHitCorrection_H
#define RecoPixelVertexing_PixelTriplets_ThirdHitCorrection_H

namespace edm {
  class EventSetup;
}
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

class DetLayer;

#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

class ThirdHitCorrection {
public:
  typedef PixelRecoRange<float> Range;

  ThirdHitCorrection() {}

  void init(const edm::EventSetup &es,
            float pt,
            const DetLayer &layer1,
            const DetLayer &layer2,
            const DetLayer &layer3,
            bool useMultipleScattering,
            bool useBendingCorrection);

  void init(const edm::EventSetup &es,
            float pt,
            const DetLayer &layer3,
            bool useMultipleScattering,
            bool useBendingCorrection);

  ThirdHitCorrection(const edm::EventSetup &es,
                     float pt,
                     const DetLayer *layer,
                     const PixelRecoLineRZ &line,
                     const PixelRecoPointRZ &constraint,
                     int ol,
                     bool useMultipleScattering,
                     bool useBendingCorrection) {
    init(es, pt, *layer, useMultipleScattering, useBendingCorrection);
    init(line, constraint, ol);
  }

  void init(const PixelRecoLineRZ &line, const PixelRecoPointRZ &constraint, int ol);

  void correctRPhiRange(Range &range) const {
    range.first -= theMultScattCorrRPhi;
    range.second += theMultScattCorrRPhi;
  }

  void correctRZRange(Range &range) const;

private:
  bool theBarrel;

  bool theUseMultipleScattering;
  bool theUseBendingCorrection;

  PixelRecoLineRZ theLine;
  float theMultScattCorrRPhi = 0;
  float theMScoeff = 0;
  float thePt;

  pixelrecoutilities::LongitudinalBendingCorrection theBendingCorrection;
  MultipleScatteringParametrisation sigmaRPhi;
};

#endif
