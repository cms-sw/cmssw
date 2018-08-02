//
// Author: Felice Pantaleo, CERN
//

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"

__global__ void
kernel_debug(unsigned int numberOfLayerPairs_, unsigned int numberOfLayers_,
             const GPULayerDoublets *gpuDoublets,
             const GPULayerHits *gpuHitsOnLayers, GPUCACell *cells,
             GPU::VecArray<unsigned int, 512> *isOuterHitOfCell,
             GPU::SimpleVector<Quadruplet> *foundNtuplets,
             float ptmin, float region_origin_x, float region_origin_y,
             float region_origin_radius, const float thetaCut,
             const float phiCut, const float hardPtCut,
             unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {
  if (threadIdx.x == 0 and blockIdx.x == 0)
    foundNtuplets->reset();

  printf("kernel_debug_create: theEvent contains numberOfLayerPairs_: %d\n",
         numberOfLayerPairs_);
  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs_;
       ++layerPairIndex) {

    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    int numberOfDoublets = gpuDoublets[layerPairIndex].size;
    printf(
        "kernel_debug_create: layerPairIndex: %d inner %d outer %d size %u\n",
        layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = outerLayerId * maxNumberOfHits_;
    printf("kernel_debug_create: theIdOfThefirstCellInLayerPair: %d "
           "globalFirstHitIdx %d\n",
           globalFirstDoubletIdx, globalFirstHitIdx);

    for (unsigned int i = 0; i < gpuDoublets[layerPairIndex].size; i++) {

      auto globalCellIdx = i + globalFirstDoubletIdx;
      auto &thisCell = cells[globalCellIdx];
      auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
      thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
                    layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region_origin_x, region_origin_y);

      isOuterHitOfCell[globalFirstHitIdx + outerHitId].push_back(
          globalCellIdx);
    }
  }

  // for(unsigned int layerIndex = 0; layerIndex < numberOfLayers_;++layerIndex )
  // {
  //     auto numberOfHitsOnLayer = gpuHitsOnLayers[layerIndex].size;
  //     for(unsigned hitId = 0; hitId < numberOfHitsOnLayer; hitId++)
  //     {
  //
  //         if(isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].size()>0)
  //         {
  //             printf("\nlayer %d hit %d is outer hit of %d
  //             cells\n",layerIndex, hitId,
  //             isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].size());
  //             printf("\n\t%f %f %f
  //             \n",gpuHitsOnLayers[layerIndex].x[hitId],gpuHitsOnLayers[layerIndex].y[hitId],gpuHitsOnLayers[layerIndex].z[hitId]);
  //
  //             for(unsigned cell = 0; cell<
  //             isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].size();
  //             cell++)
  //             {
  //                 printf("cell %d\n",
  //                 isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].m_data[cell]);
  //                 auto& thisCell =
  //                 cells[isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].m_data[cell]];
  //                             float x1, y1, z1, x2, y2, z2;
  //
  //                             x1 = thisCell.get_inner_x();
  //                             y1 = thisCell.get_inner_y();
  //                             z1 = thisCell.get_inner_z();
  //                             x2 = thisCell.get_outer_x();
  //                             y2 = thisCell.get_outer_y();
  //                             z2 = thisCell.get_outer_z();
  //                 printf("\n\tDEBUG cellid %d innerhit outerhit (xyz) (%f %f
  //                 %f), (%f %f
  //                 %f)\n",isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].m_data[cell],
  //                 x1,y1,z1,x2,y2,z2);
  //             }
  //         }
  //     }
  // }

  // starting connect

  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs_;
       ++layerPairIndex) {

    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    int numberOfDoublets = gpuDoublets[layerPairIndex].size;
    printf("kernel_debug_connect: connecting layerPairIndex: %d inner %d outer "
           "%d size %u\n",
           layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = innerLayerId * maxNumberOfHits_;
    //        printf("kernel_debug_connect: theIdOfThefirstCellInLayerPair: %d
    //        globalFirstHitIdx %d\n", globalFirstDoubletIdx,
    //        globalFirstHitIdx);

    for (unsigned int i = 0; i < numberOfDoublets; i++) {

      auto globalCellIdx = i + globalFirstDoubletIdx;

      auto &thisCell = cells[globalCellIdx];
      auto innerHitId = thisCell.get_inner_hit_id();
      auto numberOfPossibleNeighbors =
          isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
      //            if(numberOfPossibleNeighbors>0)
      //            printf("kernel_debug_connect: cell: %d has %d possible
      //            neighbors\n", globalCellIdx, numberOfPossibleNeighbors);
      float x1, y1, z1, x2, y2, z2;

      x1 = thisCell.get_inner_x();
      y1 = thisCell.get_inner_y();
      z1 = thisCell.get_inner_z();
      x2 = thisCell.get_outer_x();
      y2 = thisCell.get_outer_y();
      z2 = thisCell.get_outer_z();
      printf("\n\n\nDEBUG cellid %d innerhit outerhit (xyz) (%f %f %f), (%f %f "
             "%f)\n",
             globalCellIdx, x1, y1, z1, x2, y2, z2);

      for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
        unsigned int otherCell =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId][j];

        float x3, y3, z3, x4, y4, z4;
        x3 = cells[otherCell].get_inner_x();
        y3 = cells[otherCell].get_inner_y();
        z3 = cells[otherCell].get_inner_z();
        x4 = cells[otherCell].get_outer_x();
        y4 = cells[otherCell].get_outer_y();
        z4 = cells[otherCell].get_outer_z();

        printf("kernel_debug_connect: checking compatibility with %d \n",
               otherCell);
        printf("DEBUG \tinnerhit outerhit (xyz) (%f %f %f), (%f %f %f)\n", x3,
               y3, z3, x4, y4, z4);

        if (thisCell.check_alignment_and_tag(
                cells, otherCell, ptmin, region_origin_x, region_origin_y,
                region_origin_radius, thetaCut, phiCut, hardPtCut)) {

          printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n",
                 globalCellIdx, otherCell);

          cells[otherCell].theOuterNeighbors.push_back(globalCellIdx);
        }
      }
    }
  }
}

__global__ void debug_input_data(unsigned int numberOfLayerPairs_,
                                 const GPULayerDoublets *gpuDoublets,
                                 const GPULayerHits *gpuHitsOnLayers,
                                 float ptmin, float region_origin_x,
                                 float region_origin_y,
                                 float region_origin_radius,
                                 unsigned int maxNumberOfHits_) {
  printf("GPU: Region ptmin %f , region_origin_x %f , region_origin_y %f , "
         "region_origin_radius  %f \n",
         ptmin, region_origin_x, region_origin_y, region_origin_radius);
  printf("GPU: numberOfLayerPairs_: %d\n", numberOfLayerPairs_);

  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs_;
       ++layerPairIndex) {
    printf("\t numberOfDoublets: %d \n", gpuDoublets[layerPairIndex].size);
    printf("\t innerLayer: %d outerLayer: %d \n",
           gpuDoublets[layerPairIndex].innerLayerId,
           gpuDoublets[layerPairIndex].outerLayerId);

    for (unsigned int cellIndexInLayerPair = 0;
         cellIndexInLayerPair < gpuDoublets[layerPairIndex].size;
         ++cellIndexInLayerPair) {

      if (cellIndexInLayerPair < 5) {
        auto innerhit =
            gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair];
        auto innerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .x[innerhit];
        auto innerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .y[innerhit];
        auto innerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .z[innerhit];

        auto outerhit =
            gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair + 1];
        auto outerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .x[outerhit];
        auto outerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .y[outerhit];
        auto outerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .z[outerhit];
        printf("\t \t %d innerHit: %d %f %f %f outerHit: %d %f %f %f\n",
               cellIndexInLayerPair, innerhit, innerX, innerY, innerZ, outerhit,
               outerX, outerY, outerZ);
      }
    }
  }
}

template <int maxNumberOfQuadruplets_>
__global__ void kernel_debug_find_ntuplets(
    unsigned int numberOfRootLayerPairs_, const GPULayerDoublets *gpuDoublets,
    GPUCACell *cells,
    GPU::VecArray<Quadruplet, maxNumberOfQuadruplets_> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_) {
  printf("numberOfRootLayerPairs_ = %d", numberOfRootLayerPairs_);
  for (int rootLayerPair = 0; rootLayerPair < numberOfRootLayerPairs_;
       ++rootLayerPair) {
    unsigned int rootLayerPairIndex = rootLayerPairs[rootLayerPair];
    auto globalFirstDoubletIdx = rootLayerPairIndex * maxNumberOfDoublets_;

    GPU::VecArray<unsigned int, 3> stack;
    for (int i = 0; i < gpuDoublets[rootLayerPairIndex].size; i++) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      stack.reset();
      stack.push_back(globalCellIdx);
      cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack,
                                         minHitsPerNtuplet);
    }
    printf("found quadruplets: %d", foundNtuplets->size());
  }
}

__global__ void kernel_create(
    const unsigned int numberOfLayerPairs_, const GPULayerDoublets *gpuDoublets,
    const GPULayerHits *gpuHitsOnLayers, GPUCACell *cells,
    GPU::VecArray<unsigned int, 512> *isOuterHitOfCell,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    const float region_origin_x, const float region_origin_y,
    unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {

  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndexInLayerPair == 0 && layerPairIndex == 0) {
    foundNtuplets->reset();
  }

  if (layerPairIndex < numberOfLayerPairs_) {
    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = outerLayerId * maxNumberOfHits_;

    for (unsigned int i = cellIndexInLayerPair;
         i < gpuDoublets[layerPairIndex].size; i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      auto &thisCell = cells[globalCellIdx];
      auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
      thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
                    layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region_origin_x, region_origin_y);

      isOuterHitOfCell[globalFirstHitIdx + outerHitId].push_back(
          globalCellIdx);
    }
  }
}

__global__ void
kernel_connect(unsigned int numberOfLayerPairs_,
               const GPULayerDoublets *gpuDoublets, GPUCACell *cells,
               GPU::VecArray< unsigned int, 512> *isOuterHitOfCell,
               float ptmin, float region_origin_x, float region_origin_y,
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {
  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if (layerPairIndex < numberOfLayerPairs_) {
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = innerLayerId * maxNumberOfHits_;

    for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
         i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;

      auto &thisCell = cells[globalCellIdx];
      auto innerHitId = thisCell.get_inner_hit_id();
      auto numberOfPossibleNeighbors =
          isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
      for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
        unsigned int otherCell =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId][j];

        if (thisCell.check_alignment_and_tag(
                cells, otherCell, ptmin, region_origin_x, region_origin_y,
                region_origin_radius, thetaCut, phiCut, hardPtCut)) {
          cells[otherCell].theOuterNeighbors.push_back(globalCellIdx);
        }
      }
    }
  }
}

__global__ void kernel_find_ntuplets(
    unsigned int numberOfRootLayerPairs_, const GPULayerDoublets *gpuDoublets,
    GPUCACell *cells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_)
{
  if (blockIdx.y < numberOfRootLayerPairs_) {
    unsigned int cellIndexInRootLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rootLayerPairIndex = rootLayerPairs[blockIdx.y];
    auto globalFirstDoubletIdx = rootLayerPairIndex * maxNumberOfDoublets_;
    GPU::VecArray<unsigned int, 3> stack;
    for (int i = cellIndexInRootLayerPair;
         i < gpuDoublets[rootLayerPairIndex].size;
         i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      stack.reset();
      stack.push_back_unsafe(globalCellIdx);
      cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);
    }
  }
}

template <int maxNumberOfDoublets_>
__global__ void
kernel_print_found_ntuplets(GPU::SimpleVector<Quadruplet> *foundNtuplets) {
  for (int i = 0; i < foundNtuplets->size(); ++i) {
    printf("\nquadruplet %d: %d %d, %d %d, %d %d\n", i,
           (*foundNtuplets)[i].layerPairsAndCellId[0].x,
           (*foundNtuplets)[i].layerPairsAndCellId[0].y -
               maxNumberOfDoublets_ *
                   ((*foundNtuplets)[i].layerPairsAndCellId[0].x),
           (*foundNtuplets)[i].layerPairsAndCellId[1].x,
           (*foundNtuplets)[i].layerPairsAndCellId[1].y -
               maxNumberOfDoublets_ *
                   (*foundNtuplets)[i].layerPairsAndCellId[1].x,
           (*foundNtuplets)[i].layerPairsAndCellId[2].x,
           (*foundNtuplets)[i].layerPairsAndCellId[2].y -
               maxNumberOfDoublets_ *
                   ((*foundNtuplets)[i].layerPairsAndCellId[2].x));
  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  cudaFreeHost(h_indices_);
  cudaFreeHost(h_doublets_);
  cudaFreeHost(h_x_);
  cudaFreeHost(h_y_);
  cudaFreeHost(h_z_);
  cudaFreeHost(h_rootLayerPairs_);
  for (size_t i = 0; i < h_foundNtupletsVec_.size(); ++i)
  {
    cudaFreeHost(h_foundNtupletsVec_[i]);
    cudaFreeHost(h_foundNtupletsData_[i]);
    cudaFree(d_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsData_[i]);
  }
  cudaFreeHost(tmp_layers_);
  cudaFreeHost(tmp_layerDoublets_);
  cudaFreeHost(h_layers_);

  cudaFree(d_indices_);
  cudaFree(d_doublets_);
  cudaFree(d_layers_);
  cudaFree(d_x_);
  cudaFree(d_y_);
  cudaFree(d_z_);
  cudaFree(d_rootLayerPairs_);
  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  cudaCheck(cudaMallocHost(&h_doublets_, maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));
  cudaCheck(cudaMallocHost(&h_indices_, maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * 2 * sizeof(int)));
  cudaCheck(cudaMallocHost(&h_x_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMallocHost(&h_y_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMallocHost(&h_z_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMallocHost(&h_rootLayerPairs_, maxNumberOfRootLayerPairs_ * sizeof(int)));

  cudaCheck(cudaMalloc(&d_indices_, maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * 2 * sizeof(int)));
  cudaCheck(cudaMalloc(&d_doublets_, maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));
  cudaCheck(cudaMalloc(&d_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits)));
  cudaCheck(cudaMalloc(&d_x_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMalloc(&d_y_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMalloc(&d_z_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMalloc(&d_rootLayerPairs_, maxNumberOfRootLayerPairs_ * sizeof(unsigned int)));

  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * sizeof(GPUCACell)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));

  h_foundNtupletsVec_.resize(maxNumberOfRegions_);
  h_foundNtupletsData_.resize(maxNumberOfRegions_);
  d_foundNtupletsVec_.resize(maxNumberOfRegions_);
  d_foundNtupletsData_.resize(maxNumberOfRegions_);

  // FIXME this could be rewritten with a single pair of cudaMallocHost / cudaMalloc
  for (int i = 0; i < maxNumberOfRegions_; ++i) {
    cudaCheck(cudaMallocHost(&h_foundNtupletsData_[i],  sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMallocHost(&h_foundNtupletsVec_[i],   sizeof(GPU::SimpleVector<Quadruplet>)));
    new(h_foundNtupletsVec_[i]) GPU::SimpleVector<Quadruplet>(maxNumberOfQuadruplets_, h_foundNtupletsData_[i]);
    cudaCheck(cudaMalloc(&d_foundNtupletsData_[i],      sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMemset(d_foundNtupletsData_[i], 0x00, sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMalloc(&d_foundNtupletsVec_[i],       sizeof(GPU::SimpleVector<Quadruplet>)));
    GPU::SimpleVector<Quadruplet> tmp_foundNtuplets(maxNumberOfQuadruplets_, d_foundNtupletsData_[i]);
    cudaCheck(cudaMemcpy(d_foundNtupletsVec_[i], & tmp_foundNtuplets, sizeof(GPU::SimpleVector<Quadruplet>), cudaMemcpyDefault));
  }

  cudaCheck(cudaMallocHost(&tmp_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits)));
  cudaCheck(cudaMallocHost(&tmp_layerDoublets_,maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));
  cudaCheck(cudaMallocHost(&h_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits)));
}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, cudaStream_t cudaStream)
{
  assert(regionIndex < maxNumberOfRegions_);
  dim3 numberOfBlocks_create(64, numberOfLayerPairs_);
  dim3 numberOfBlocks_connect(32, numberOfLayerPairs_);
  dim3 numberOfBlocks_find(16, numberOfRootLayerPairs_);
  h_foundNtupletsVec_[regionIndex]->reset();
  kernel_create<<<numberOfBlocks_create, 32, 0, cudaStream>>>(
      numberOfLayerPairs_, d_doublets_, d_layers_, device_theCells_,
      device_isOuterHitOfCell_, d_foundNtupletsVec_[regionIndex],
      region.origin().x(), region.origin().y(), maxNumberOfDoublets_,
      maxNumberOfHits_);
  cudaCheck(cudaGetLastError());

  kernel_connect<<<numberOfBlocks_connect, 512, 0, cudaStream>>>(
      numberOfLayerPairs_, d_doublets_, device_theCells_,
      device_isOuterHitOfCell_,
      region.ptMin(), region.origin().x(), region.origin().y(),
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets_, maxNumberOfHits_);
  cudaCheck(cudaGetLastError());

  kernel_find_ntuplets<<<numberOfBlocks_find, 1024, 0, cudaStream>>>(
      numberOfRootLayerPairs_, d_doublets_, device_theCells_,
      d_foundNtupletsVec_[regionIndex],
      d_rootLayerPairs_, 4, maxNumberOfDoublets_);
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaMemcpyAsync(h_foundNtupletsVec_[regionIndex], d_foundNtupletsVec_[regionIndex],
                            sizeof(GPU::SimpleVector<Quadruplet>),
                            cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(h_foundNtupletsData_[regionIndex], d_foundNtupletsData_[regionIndex],
                            maxNumberOfQuadruplets_*sizeof(Quadruplet),
                            cudaMemcpyDeviceToHost, cudaStream));

}

std::vector<std::array<std::pair<int, int>, 3>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex, cudaStream_t cudaStream)
{
  h_foundNtupletsVec_[regionIndex]->set_data(h_foundNtupletsData_[regionIndex]);
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>),
                            cudaStream));
  std::vector<std::array<std::pair<int, int>, 3>> quadsInterface;
  for (int i = 0; i < h_foundNtupletsVec_[regionIndex]->size(); ++i) {
    auto const& layerPairsAndCellId = (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId;
    std::array<std::pair<int, int>, 3> tmpQuad = {
        {std::make_pair(layerPairsAndCellId[0].x, layerPairsAndCellId[0].y - maxNumberOfDoublets_ * layerPairsAndCellId[0].x),
         std::make_pair(layerPairsAndCellId[1].x, layerPairsAndCellId[1].y - maxNumberOfDoublets_ * layerPairsAndCellId[1].x),
         std::make_pair(layerPairsAndCellId[2].x, layerPairsAndCellId[2].y - maxNumberOfDoublets_ * layerPairsAndCellId[2].x)}};

    quadsInterface.push_back(tmpQuad);
  }
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const & hh, float phicut, cudaStream_t stream) {
   auto nhits = hh.nHits;

  float phiCut=0.06;
  int threadsPerBlock = 256;
  int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;

  gpuPixelDoublets::getDoubletsFromHisto<<<blocks, threadsPerBlock, 0, stream>>>(hh.gpu_d,phiCut);
  cudaCheck(cudaGetLastError());
}
