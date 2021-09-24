#include "ThirdHitCorrection.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"

using namespace pixelrecoutilities;

namespace {
  template <class T>
  inline T sqr(T t) {
    return t * t;
  }
}  // namespace

using pixelrecoutilities::LongitudinalBendingCorrection;

void ThirdHitCorrection::init(float pt,
                              const DetLayer &layer3,
                              bool useMultipleScattering,
                              const MultipleScatteringParametrisationMaker *msmaker,
                              bool useBendingCorrection,
                              const MagneticField *bfield) {
  theUseMultipleScattering = useMultipleScattering;
  theUseBendingCorrection = useBendingCorrection;
  if (useBendingCorrection) {
    theBendingCorrection.init(pt, *bfield);
  }

  theMultScattCorrRPhi = 0;
  theMScoeff = 0;

  theBarrel = layer3.isBarrel();
  thePt = pt;

  if (theUseMultipleScattering)
    sigmaRPhi = msmaker->parametrisation(&layer3);
}

void ThirdHitCorrection::init(float pt,
                              const DetLayer &layer1,
                              const DetLayer &layer2,
                              const DetLayer &layer3,
                              bool useMultipleScattering,
                              const MultipleScatteringParametrisationMaker *msmaker,
                              bool useBendingCorrection,
                              const MagneticField *bfield) {
  init(pt, layer3, useMultipleScattering, msmaker, useBendingCorrection, bfield);

  if (!theUseMultipleScattering)
    return;

  auto point3 = [&]() -> PixelRecoPointRZ {
    if (theBarrel) {
      const BarrelDetLayer &bl = static_cast<const BarrelDetLayer &>(layer3);
      float rLayer = bl.specificSurface().radius();
      auto zmax = 0.5f * layer3.surface().bounds().length();
      return PixelRecoPointRZ(rLayer, zmax);
    } else {
      const ForwardDetLayer &fl = static_cast<const ForwardDetLayer &>(layer3);
      auto maxR = fl.specificSurface().outerRadius();
      auto layerZ = layer3.position().z();
      return PixelRecoPointRZ(maxR, layerZ);
    }
  };

  PixelRecoPointRZ zero(0., 0.);
  SimpleLineRZ line(zero, point3());

  auto point2 = [&]() -> PixelRecoPointRZ {
    if (layer2.isBarrel()) {
      const BarrelDetLayer &bl = static_cast<const BarrelDetLayer &>(layer2);
      float rLayer = bl.specificSurface().radius();
      return PixelRecoPointRZ(rLayer, line.zAtR(rLayer));
    } else {
      auto layerZ = layer2.position().z();
      return PixelRecoPointRZ(line.rAtZ(layerZ), layerZ);
    }
  };

  theMultScattCorrRPhi = 3.f * (*sigmaRPhi)(pt, line.cotLine(), point2(), layer2.seqNum());
}

void ThirdHitCorrection::init(const PixelRecoLineRZ &line, const PixelRecoPointRZ &constraint, int il) {
  theLine = line;
  if (!theUseMultipleScattering)
    return;

  // auto newCorr = theMultScattCorrRPhi;
  theMultScattCorrRPhi = 3.f * (*sigmaRPhi)(thePt, line.cotLine(), constraint, il);
  // std::cout << "ThirdHitCorr " << (theBarrel ? "B " : "F " )<< theMultScattCorrRPhi << ' ' << newCorr << ' ' << newCorr/theMultScattCorrRPhi << std::endl;
  float overSinTheta = std::sqrt(1.f + sqr(line.cotLine()));
  if (theBarrel) {
    theMScoeff = theMultScattCorrRPhi * overSinTheta;
  } else {
    float overCosTheta = std::abs(line.cotLine()) < 1.e-4f ? 1.e4f : overSinTheta / std::abs(line.cotLine());
    theMScoeff = theMultScattCorrRPhi * overCosTheta;
  }
}

void ThirdHitCorrection::correctRZRange(Range &range) const {
  range.first -= theMScoeff;
  range.second += theMScoeff;

  if (theUseBendingCorrection) {
    if (theBarrel) {
      float cotTheta = theLine.cotLine();
      if (cotTheta > 0) {
        float radius = theLine.rAtZ(range.max());
        float corr = theBendingCorrection(radius) * cotTheta;
        range.second += corr;
      } else {
        float radius = theLine.rAtZ(range.min());
        float corr = theBendingCorrection(radius) * std::abs(cotTheta);
        range.first -= corr;
      }
    } else {
      float radius = range.max();
      float corr = theBendingCorrection(radius);
      range.first -= corr;
    }
  }
}
