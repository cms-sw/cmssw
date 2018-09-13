//
// Author: Felice Pantaleo, CERN
//

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"
#include <cstdint>

using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

__global__ void
kernel_checkOverflows(GPU::SimpleVector<Quadruplet> *foundNtuplets,
               GPUCACell *cells, uint32_t const * nCells,
               GPU::VecArray< unsigned int, 256> *isOuterHitOfCell,
               uint32_t nHits) {

 auto idx = threadIdx.x + blockIdx.x * blockDim.x;
 #ifdef GPU_DEBUG
 if (0==idx)
   printf("number of found cells %d\n",*nCells);
 #endif
 if (idx < (*nCells) ) {
   auto &thisCell = cells[idx];
   if (thisCell.theOuterNeighbors.full()) //++tooManyNeighbors[thisCell.theLayerPairId];
     printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
 }
 if (idx < nHits) {
   if (isOuterHitOfCell[idx].full()) // ++tooManyOuterHitOfCell;
     printf("OuterHitOfCell overflow %d\n", idx); 
 }

}


__global__ void
kernel_connect(GPU::SimpleVector<Quadruplet> *foundNtuplets,
               GPUCACell *cells, uint32_t const * nCells,
               GPU::VecArray< unsigned int, 256> *isOuterHitOfCell,
               float ptmin, 
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {

  float region_origin_x = 0.;
  float region_origin_y = 0.;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (0==cellIndex) foundNtuplets->reset(); // ready for next kernel

  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
     auto otherCell = isOuterHitOfCell[innerHitId][j];

     if (thisCell.check_alignment_and_tag(
                 cells, otherCell, ptmin, region_origin_x, region_origin_y,
                  region_origin_radius, thetaCut, phiCut, hardPtCut)
        ) {
          cells[otherCell].theOuterNeighbors.push_back(cellIndex);
     }
  }
}

__global__ void kernel_find_ntuplets(
    GPUCACell *cells, uint32_t const * nCells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_)
{

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];
  if (thisCell.theLayerPairId!=0 && thisCell.theLayerPairId!=3 && thisCell.theLayerPairId!=8) return; // inner layer is 0 FIXME
  GPU::VecArray<unsigned int, 3> stack;
  stack.reset();
  thisCell.find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);
  assert(stack.size()==0);
  // printf("in %d found quadruplets: %d\n", cellIndex, foundNtuplets->size());
}

__global__ void
kernel_print_found_ntuplets(GPU::SimpleVector<Quadruplet> *foundNtuplets, int maxPrint) {
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->size()); ++i) {
    printf("\nquadruplet %d: %d %d %d %d\n", i,
           (*foundNtuplets)[i].hitId[0],
           (*foundNtuplets)[i].hitId[1],
           (*foundNtuplets)[i].hitId[2],
           (*foundNtuplets)[i].hitId[3]
          );
         
  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  for (size_t i = 0; i < h_foundNtupletsVec_.size(); ++i)
  {
    cudaFreeHost(h_foundNtupletsVec_[i]);
    cudaFreeHost(h_foundNtupletsData_[i]);
    cudaFree(d_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsData_[i]);
  }

  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
  cudaFree(device_nCells_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * sizeof(GPUCACell)));
  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));
  cudaCheck(cudaMemset(device_nCells_, 0, sizeof(uint32_t)));

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

}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, HitsOnCPU const & hh,
                                                bool transferToCPU,
                                                cudaStream_t cudaStream)
{
  assert(regionIndex < maxNumberOfRegions_);
  assert(0==regionIndex);

  h_foundNtupletsVec_[regionIndex]->reset();

  auto nhits = hh.nHits;

  auto numberOfBlocks = (maxNumberOfDoublets_ + 512 - 1)/512;
  kernel_connect<<<numberOfBlocks, 512, 0, cudaStream>>>(
      d_foundNtupletsVec_[regionIndex], // needed only to be reset, ready for next kernel
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      region.ptMin(), 
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets_, maxNumberOfHits_
  );
  cudaCheck(cudaGetLastError());

  kernel_find_ntuplets<<<numberOfBlocks, 512, 0, cudaStream>>>(
      device_theCells_, device_nCells_,
      d_foundNtupletsVec_[regionIndex],
      4, maxNumberOfDoublets_);
  cudaCheck(cudaGetLastError());


  numberOfBlocks = (std::max(int(nhits), maxNumberOfDoublets_) + 512 - 1)/512;
  kernel_checkOverflows<<<numberOfBlocks, 512, 0, cudaStream>>>(
                        d_foundNtupletsVec_[regionIndex],
                        device_theCells_, device_nCells_,
                        device_isOuterHitOfCell_, nhits
                       );


  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(d_foundNtupletsVec_[regionIndex], 10);

  if(transferToCPU) {
    cudaCheck(cudaMemcpyAsync(h_foundNtupletsVec_[regionIndex], d_foundNtupletsVec_[regionIndex],
                              sizeof(GPU::SimpleVector<Quadruplet>),
                              cudaMemcpyDeviceToHost, cudaStream));

    cudaCheck(cudaMemcpyAsync(h_foundNtupletsData_[regionIndex], d_foundNtupletsData_[regionIndex],
                              maxNumberOfQuadruplets_*sizeof(Quadruplet),
                              cudaMemcpyDeviceToHost, cudaStream));
  }
}

void CAHitQuadrupletGeneratorGPU::cleanup(cudaStream_t cudaStream) {
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>),
                            cudaStream));
  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), cudaStream));
}

std::vector<std::array<int, 4>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex)
{
  assert(0==regionIndex);
  h_foundNtupletsVec_[regionIndex]->set_data(h_foundNtupletsData_[regionIndex]);

  std::vector<std::array<int, 4>> quadsInterface(h_foundNtupletsVec_[regionIndex]->size());
  for (int i = 0; i < h_foundNtupletsVec_[regionIndex]->size(); ++i) {
    for (int j = 0; j<4; ++j) quadsInterface[i][j] = (*h_foundNtupletsVec_[regionIndex])[i].hitId[j];
  }
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
  auto nhits = hh.nHits;

  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize;
  int blocks = (3 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  gpuPixelDoublets::getDoubletsFromHisto<<<blocks, threadsPerBlock, 0, stream>>>(device_theCells_, device_nCells_, hh.gpu_d, device_isOuterHitOfCell_);
  cudaCheck(cudaGetLastError());
}
