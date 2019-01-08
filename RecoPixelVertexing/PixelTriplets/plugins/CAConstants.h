#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cuda_runtime.h>
#include <cstdint>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelTrackingGPUConstants.h"


namespace CAConstants {

   // constants
   constexpr uint32_t maxNumberOfQuadruplets() { return 10000; }
   constexpr uint32_t maxCellsPerHit() { return 128; }
   constexpr uint32_t maxNumberOfLayerPairs() { return 13; }
   constexpr uint32_t maxNumberOfLayers() { return 10; }
   constexpr uint32_t maxNumberOfDoublets() { return 262144; }
   constexpr uint32_t maxTuples() { return 10000;}

   // types
   using hindex_type = uint16_t; // FIXME from siPixelRecHitsHeterogeneousProduct
   using tindex_type = uint16_t; //  for tuples
   using OuterHitOfCell = GPU::VecArray< uint32_t, maxCellsPerHit()>;
   using TuplesContainer = OneToManyAssoc<hindex_type, maxTuples(), 5*maxTuples()>;
   using HitToTuple = OneToManyAssoc<tindex_type, PixelGPUConstants::maxNumberOfHits, 4*maxTuples()>; // 3.5 should be enough

}



#endif

