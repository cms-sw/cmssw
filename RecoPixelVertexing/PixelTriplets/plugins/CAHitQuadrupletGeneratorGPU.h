#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h

#include <cuda_runtime.h>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelTrackingGPUConstants.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/RecHitsMap.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircleFit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/RecHitsMap.h"

#include "CAHitQuadrupletGeneratorKernels.h"
#include "RiemannFitOnGPU.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

// FIXME  (split header???)
#include "GPUCACell.h"

class TrackingRegion;

namespace edm {
    class Event;
    class EventSetup;
    class ParameterSetDescription;
}

class CAHitQuadrupletGeneratorGPU {
public:

    using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
    using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
    using hindex_type = siPixelRecHitsHeterogeneousProduct::hindex_type;

    using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;
    using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;
    using Quality = pixelTuplesHeterogeneousProduct::Quality;
    using Output = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;

    static constexpr unsigned int minLayers = 4;
    using  ResultType = OrderedHitSeeds;

public:

    CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC): CAHitQuadrupletGeneratorGPU(cfg, iC) {}
    CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

    ~CAHitQuadrupletGeneratorGPU();

    static void fillDescriptions(edm::ParameterSetDescription& desc);
    static const char *fillDescriptionsLabel() { return "caHitQuadrupletGPU"; }

    void initEvent(const edm::Event& ev, const edm::EventSetup& es);

    void buildDoublets(HitsOnCPU const & hh, cudaStream_t stream);

    void hitNtuplets(HitsOnCPU const & hh,
                     const edm::EventSetup& es,
                     bool doRiemannFit,
                     bool transferToCPU,
                     cudaStream_t stream);

    TuplesOnCPU getOutput() const {
       return TuplesOnCPU { std::move(indToEdm), hitsOnCPU->gpu_d, tuples_,  helix_fit_results_, quality_, gpu_d, nTuples_};
    }

    void cleanup(cudaStream_t stream);
    void fillResults(const TrackingRegion &region, SiPixelRecHitCollectionNew const & rechits,
                     std::vector<OrderedHitSeeds>& result,
                     const edm::EventSetup& es);

    void allocateOnGPU();
    void deallocateOnGPU();

private:

    void launchKernels(HitsOnCPU const & hh, bool doRiemannFit, bool transferToCPU, cudaStream_t);


    std::vector<std::array<int,4>> fetchKernelResult(int);


    CAHitQuadrupletGeneratorKernels kernels;
    RiemannFitOnGPU fitter;

    // not really used at the moment
    const float caThetaCut = 0.00125f;
    const float caPhiCut = 0.1f;
    const float caHardPtCut = 0.f;


    // products
    std::vector<uint32_t> indToEdm; // index of    tuple in reco tracks....
    TuplesOnGPU * gpu_d = nullptr;   // copy of the structure on the gpu itself: this is the "Product"
    TuplesOnGPU::Container * tuples_ = nullptr;
    Rfit::helix_fit * helix_fit_results_ = nullptr;
    Quality * quality_ =  nullptr;
    uint32_t nTuples_ = 0;
    TuplesOnGPU gpu_;

    // input
    HitsOnCPU const * hitsOnCPU=nullptr;

    RecHitsMap<TrackingRecHit const *> hitmap_ = RecHitsMap<TrackingRecHit const *>(nullptr);

};

#endif // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h
