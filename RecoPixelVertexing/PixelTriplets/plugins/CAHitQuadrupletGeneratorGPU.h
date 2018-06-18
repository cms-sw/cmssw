#ifndef RECOPIXELVERTEXING_PIXELTRIPLETS_CAHITQUADRUPLETGENERATORGPU_H
#define RECOPIXELVERTEXING_PIXELTRIPLETS_CAHITQUADRUPLETGENERATORGPU_H

#include <cuda_runtime.h>
#include "GPUHitsAndDoublets.h"

#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircleFit.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "CAGraph.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "GPUCACell.h"

class TrackingRegion;
class SeedingLayerSetsHits;

namespace edm {
    class Event;
    class EventSetup;
    class ParameterSetDescription;
}

class CAHitQuadrupletGeneratorGPU {
public:
    typedef LayerHitMapCache LayerCacheType;

    static constexpr unsigned int minLayers = 4;
    typedef OrderedHitSeeds ResultType;

public:

    CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC): CAHitQuadrupletGeneratorGPU(cfg, iC) {}
    CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

    ~CAHitQuadrupletGeneratorGPU();

    static void fillDescriptions(edm::ParameterSetDescription& desc);
    static const char *fillDescriptionsLabel() { return "caHitQuadrupletGPU"; }

    void initEvent(const edm::Event& ev, const edm::EventSetup& es);

    void hitNtuplets(const IntermediateHitDoublets& regionDoublets,
                     const edm::EventSetup& es,
                     const SeedingLayerSetsHits& layers, cudaStream_t stream);
    void fillResults(const IntermediateHitDoublets& regionDoublets,
                     std::vector<OrderedHitSeeds>& result,
                     const edm::EventSetup& es,
                     const SeedingLayerSetsHits& layers, cudaStream_t stream);

    void allocateOnGPU();
    void deallocateOnGPU();

private:
    LayerCacheType theLayerCache;

    std::unique_ptr<SeedComparitor> theComparitor;

    class QuantityDependsPtEval {
    public:

        QuantityDependsPtEval(float v1, float v2, float c1, float c2) :
        value1_(v1), value2_(v2), curvature1_(c1), curvature2_(c2) {
        }

        float value(float curvature) const {
            if (value1_ == value2_) // not enabled
                return value1_;

            if (curvature1_ < curvature)
                return value1_;
            if (curvature2_ < curvature && curvature <= curvature1_)
                return value2_ + (curvature - curvature2_) / (curvature1_ - curvature2_) * (value1_ - value2_);
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

        explicit QuantityDependsPt(const edm::ParameterSet& pset) :
        value1_(pset.getParameter<double>("value1")),
        value2_(pset.getParameter<double>("value2")),
        pt1_(pset.getParameter<double>("pt1")),
        pt2_(pset.getParameter<double>("pt2")),
        enabled_(pset.getParameter<bool>("enabled")) {
            if (enabled_ && pt1_ >= pt2_)
                throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt1 (" << pt1_ << ") needs to be smaller than pt2 (" << pt2_ << ")";
            if (pt1_ <= 0)
                throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt1 needs to be > 0; is " << pt1_;
            if (pt2_ <= 0)
                throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt2 needs to be > 0; is " << pt2_;
        }

        QuantityDependsPtEval evaluator(const edm::EventSetup& es) const {
            if (enabled_) {
                return QuantityDependsPtEval(value1_, value2_,
                        PixelRecoUtilities::curvature(1.f / pt1_, es),
                        PixelRecoUtilities::curvature(1.f / pt2_, es));
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

    void  launchKernels(const TrackingRegion &, int, cudaStream_t);
    std::vector<std::array<std::pair<int,int> ,3>> fetchKernelResult(int, cudaStream_t);
    const float extraHitRPhitolerance;
    std::vector<std::vector<const HitDoublets *>> hitDoublets;

    const QuantityDependsPt maxChi2;
    const bool fitFastCircle;
    const bool fitFastCircleChi2Cut;
    const bool useBendingCorrection;

    const float caThetaCut = 0.00125f;
    const float caPhiCut = 0.1f;
    const float caHardPtCut = 0.f;

    static constexpr int maxNumberOfQuadruplets_ = 10000;
    static constexpr int maxCellsPerHit_ = 512;
    static constexpr int maxNumberOfLayerPairs_ = 13;
    static constexpr unsigned int maxNumberOfRootLayerPairs_ = 13;
    static constexpr int maxNumberOfLayers_ = 10;
    static constexpr int maxNumberOfDoublets_ = 262144;
    static constexpr int maxNumberOfHits_ = 10000;
    static constexpr int maxNumberOfRegions_ = 30;

    unsigned int numberOfRootLayerPairs_ = 0;
    unsigned int numberOfLayerPairs_ = 0;
    unsigned int numberOfLayers_ = 0;

    GPULayerDoublets* h_doublets_;
    GPULayerHits* h_layers_;

    unsigned int* h_indices_;
    float *h_x_, *h_y_, *h_z_;
    float *d_x_, *d_y_, *d_z_;
    unsigned int* d_rootLayerPairs_;
    GPULayerHits* d_layers_;
    GPULayerDoublets* d_doublets_;
    unsigned int* d_indices_;
    unsigned int* h_rootLayerPairs_;
    std::vector<GPU::SimpleVector<Quadruplet>*> h_foundNtupletsVec_;
    std::vector<Quadruplet*> h_foundNtupletsData_;

    std::vector<GPU::SimpleVector<Quadruplet>*> d_foundNtupletsVec_;
    std::vector<Quadruplet*> d_foundNtupletsData_;

    GPUCACell* device_theCells_;
    GPU::VecArray< unsigned int, maxCellsPerHit_>* device_isOuterHitOfCell_;

    GPULayerHits* tmp_layers_;
    GPULayerDoublets* tmp_layerDoublets_;
};

#endif
