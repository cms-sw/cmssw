#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h


#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

#include "GPUCACell.h"

class CAHitQuadrupletGeneratorKernels {
public:

   using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
   using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

   using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

   using HitToTuple = CAConstants::HitToTuple;
   using TupleMultiplicity = CAConstants::TupleMultiplicity;

   CAHitQuadrupletGeneratorKernels(uint32_t minHitsPerNtuplet,
    bool earlyFishbone, bool lateFishbone) :
    minHitsPerNtuplet_(minHitsPerNtuplet),
    earlyFishbone_(earlyFishbone),
    lateFishbone_(lateFishbone){}
   ~CAHitQuadrupletGeneratorKernels() { deallocateOnGPU();}


   TupleMultiplicity const * tupleMultiplicity() const { return device_tupleMultiplicity_;}

   void launchKernels(HitsOnCPU const & hh, TuplesOnGPU & tuples_d, cudaStream_t cudaStream);

   void classifyTuples(HitsOnCPU const & hh, TuplesOnGPU & tuples_d, cudaStream_t cudaStream);

   void buildDoublets(HitsOnCPU const & hh, cudaStream_t stream);
   void allocateOnGPU();
   void deallocateOnGPU();
   void cleanup(cudaStream_t cudaStream);

private:

    // workspace
    GPUCACell* device_theCells_ = nullptr;
    GPUCACell::OuterHitOfCell* device_isOuterHitOfCell_ = nullptr;
    uint32_t* device_nCells_ = nullptr;

    HitToTuple * device_hitToTuple_ = nullptr;
    AtomicPairCounter * device_hitToTuple_apc_ = nullptr;

    TupleMultiplicity * device_tupleMultiplicity_ = nullptr;
    uint8_t * device_tmws_ = nullptr;    

    const uint32_t minHitsPerNtuplet_;
    const bool earlyFishbone_;
    const bool lateFishbone_;

};

#endif // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
