#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "CombinedHitQuadrupletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <utility>
#include <vector>

class SeedComparitor;

class PixelQuadrupletGenerator : public HitQuadrupletGeneratorFromTripletAndLayers {

typedef CombinedHitQuadrupletGenerator::LayerCacheType       LayerCacheType;

public:
  PixelQuadrupletGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC): PixelQuadrupletGenerator(cfg, iC) {}
  PixelQuadrupletGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~PixelQuadrupletGenerator() override;

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  void hitQuadruplets( const TrackingRegion& region, OrderedHitSeeds& result,
                               const edm::Event & ev, const edm::EventSetup& es,
                               const SeedingLayerSetsHits::SeedingLayerSet& tripletLayers,
                               const std::vector<SeedingLayerSetsHits::SeedingLayer>& fourthLayers) override;

  void hitQuadruplets( const TrackingRegion& region, OrderedHitSeeds& result,
                       const edm::Event& ev, const edm::EventSetup& es,
                       OrderedHitTriplets::const_iterator tripletsBegin,
                       OrderedHitTriplets::const_iterator tripletsEnd,
                       const std::vector<SeedingLayerSetsHits::SeedingLayer>& fourthLayers,
                       LayerCacheType& layerCache);

private:
  std::unique_ptr<SeedComparitor> theComparitor;

  class QuantityDependsPtEval {
  public:
    QuantityDependsPtEval(float v1, float v2, float c1, float c2):
      value1_(v1), value2_(v2), curvature1_(c1), curvature2_(c2)
    {}

    static void fillDescriptions(edm::ParameterSetDescription& desc);

    float value(float curvature) const {
      if(value1_ == value2_) // not enabled
        return value1_;

      if(curvature1_ < curvature)
        return value1_;
      if(curvature2_ < curvature && curvature <= curvature1_)
        return value2_ + (curvature-curvature2_)/(curvature1_-curvature2_) * (value1_-value2_);
      return value2_;
    }

  private:
    const float value1_;
    const float value2_;
    const float curvature1_;
    const float curvature2_;
  };

  // Linear interpolation (in curvature) between value1 at pt1 and
  // value2 at pt2. If disabled, value2 is given (the point is to
  // allow larger/smaller values of the quantity at low pt, so it
  // makes more sense to have the high-pt value as the default).
  class QuantityDependsPt {
  public:
    explicit QuantityDependsPt(const edm::ParameterSet& pset):
      value1_(pset.getParameter<double>("value1")),
      value2_(pset.getParameter<double>("value2")),
      pt1_(pset.getParameter<double>("pt1")),
      pt2_(pset.getParameter<double>("pt2")),
      enabled_(pset.getParameter<bool>("enabled"))
    {
      if(enabled_ && pt1_ >= pt2_)
        throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt1 (" << pt1_ << ") needs to be smaller than pt2 (" << pt2_ << ")";
      if(pt1_ <= 0)
        throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt1 needs to be > 0; is " << pt1_;
      if(pt2_ <= 0)
        throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt2 needs to be > 0; is " << pt2_;
    }

    QuantityDependsPtEval evaluator(const edm::EventSetup& es) const {
      if(enabled_) {
        return QuantityDependsPtEval(value1_, value2_,
                                     PixelRecoUtilities::curvature(1.f/pt1_, es),
                                     PixelRecoUtilities::curvature(1.f/pt2_, es));
      }
      return QuantityDependsPtEval(value2_, value2_, 0.f, 0.f);
    }

  private:
    const float value1_;
    const float value2_;
    const float pt1_;
    const float pt2_;
    const bool enabled_;
  };

  const float extraHitRZtolerance;
  const float extraHitRPhitolerance;
  const QuantityDependsPt extraPhiTolerance;
  const QuantityDependsPt maxChi2;
  const bool fitFastCircle;
  const bool fitFastCircleChi2Cut;
  const bool useBendingCorrection;
};



