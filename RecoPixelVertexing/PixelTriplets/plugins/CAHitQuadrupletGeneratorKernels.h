#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h


#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

#include "GPUCACell.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"



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
      unsigned long long nEmptyCells;
      unsigned long long nZeroTrackCells;
    };

   using HitsOnGPU = TrackingRecHit2DSOAView;
   using HitsOnCPU = TrackingRecHit2DCUDA;

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

   void buildDoublets(HitsOnCPU const & hh, cuda::stream_t<>& stream);
   void allocateOnGPU();
   void deallocateOnGPU();
   void cleanup(cudaStream_t cudaStream);
   void printCounters() const;

private:

   Counters * counters_ = nullptr;

   // workspace
   CAConstants::CellNeighborsVector * device_theCellNeighbors_ = nullptr;
   cudautils::device::unique_ptr<CAConstants::CellNeighbors[]> device_theCellNeighborsContainer_; 
   CAConstants::CellTracksVector * device_theCellTracks_ = nullptr;
   cudautils::device::unique_ptr<CAConstants::CellTracks[]> device_theCellTracksContainer_;


   cudautils::device::unique_ptr<GPUCACell[]> device_theCells_;
   cudautils::device::unique_ptr<GPUCACell::OuterHitOfCell[]> device_isOuterHitOfCell_;
   uint32_t* device_nCells_ = nullptr;

   HitToTuple * device_hitToTuple_ = nullptr;
   AtomicPairCounter * device_hitToTuple_apc_ = nullptr;

   TupleMultiplicity * device_tupleMultiplicity_ = nullptr;
   uint8_t * device_tmws_ = nullptr;    

   // params
   const uint32_t minHitsPerNtuplet_;
   const bool earlyFishbone_;
   const bool lateFishbone_;
   const bool idealConditions_;
   const bool doStats_;
};

#endif // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
