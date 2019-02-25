#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cuda_runtime.h>
#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelTrackingGPUConstants.h"

// #define ONLY_PHICUT

namespace CAConstants {

   // constants

   constexpr uint32_t maxNumberOfQuadruplets() { return 6*1024; }
#ifndef ONLY_PHICUT
   constexpr uint32_t maxNumberOfDoublets() { return 262144; }
   constexpr uint32_t maxCellsPerHit() { return 128; }
#else
   constexpr uint32_t maxNumberOfDoublets() { return 6*262144; }
   constexpr uint32_t maxCellsPerHit() { return 4*128; }
#endif
   constexpr uint32_t maxNumberOfLayerPairs() { return 13; }
   constexpr uint32_t maxNumberOfLayers() { return 10; }
   constexpr uint32_t maxTuples() { return 6*1024;}

   // types
   using hindex_type = uint16_t; // FIXME from siPixelRecHitsHeterogeneousProduct
   using tindex_type = uint16_t; //  for tuples
   using OuterHitOfCell = GPU::VecArray< uint32_t, maxCellsPerHit()>;
   using TuplesContainer = OneToManyAssoc<hindex_type, maxTuples(), 5*maxTuples()>;
   using HitToTuple = OneToManyAssoc<tindex_type, PixelGPUConstants::maxNumberOfHits, 4*maxTuples()>; // 3.5 should be enough
   using TupleMultiplicity = OneToManyAssoc<tindex_type,8,maxTuples()>;

}



#endif

