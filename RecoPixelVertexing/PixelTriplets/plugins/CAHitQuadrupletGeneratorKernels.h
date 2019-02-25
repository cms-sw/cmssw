#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h


#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

#include "GPUCACell.h"




class CAHitQuadrupletGeneratorKernels {
public:

    // counters
    struct Counters {
      unsigned long long nEvents;
      unsigned long long nHits;
      unsigned long long nCells;
      unsigned long long nTuples;
      unsigned long long nGoodTracks;
      unsigned long long nUsedHits;
      unsigned long long nDupHits;
      unsigned long long nKilledCells;
    };

   using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
   using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

   using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

   using HitToTuple = CAConstants::HitToTuple;
   using TupleMultiplicity = CAConstants::TupleMultiplicity;

   CAHitQuadrupletGeneratorKernels(uint32_t minHitsPerNtuplet,
    bool earlyFishbone, bool lateFishbone, 
    bool idealConditions, bool doStats) :
    minHitsPerNtuplet_(minHitsPerNtuplet),
    earlyFishbone_(earlyFishbone),
    lateFishbone_(lateFishbone),
    idealConditions_(idealConditions),
    doStats_(doStats){}
   ~CAHitQuadrupletGeneratorKernels() { deallocateOnGPU();}


   TupleMultiplicity const * tupleMultiplicity() const { return device_tupleMultiplicity_;}

   void launchKernels(HitsOnCPU const & hh, TuplesOnGPU & tuples_d, cudaStream_t cudaStream);

   void classifyTuples(HitsOnCPU const & hh, TuplesOnGPU & tuples_d, cudaStream_t cudaStream);

   void buildDoublets(HitsOnCPU const & hh, cudaStream_t stream);
   void allocateOnGPU();
   void deallocateOnGPU();
   void cleanup(cudaStream_t cudaStream);
   void printCounters() const;

private:

    Counters * counters_ = nullptr;

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
    const bool idealConditions_;
    const bool doStats_;
};

#endif // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
