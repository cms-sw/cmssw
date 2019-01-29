//
// Author: Felice Pantaleo, CERN
//

#include "CAHitQuadrupletGeneratorKernels.h"
#include <cstdint>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"
#include "gpuFishbone.h"
#include "CAConstants.h"

using namespace gpuPixelDoublets;

using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;
using Quality = pixelTuplesHeterogeneousProduct::Quality;



__global__
void kernel_checkOverflows(TuplesOnGPU::Container * foundNtuplets, AtomicPairCounter * apc,
               GPUCACell const * __restrict__ cells, uint32_t const * __restrict__ nCells,
               GPUCACell::OuterHitOfCell const * __restrict__ isOuterHitOfCell,
               uint32_t nHits) {

 __shared__ uint32_t killedCell;
 killedCell=0;
 __syncthreads();
  
 auto idx = threadIdx.x + blockIdx.x * blockDim.x;
 #ifdef GPU_DEBUG
 if (0==idx) {
   printf("number of found cells %d, found tuples %d with total hits %d,%d\n",*nCells, apc->get().m, foundNtuplets->size(), apc->get().n);
   assert(foundNtuplets->size(apc->get().m)==0);
   assert(foundNtuplets->size()==apc->get().n);
 }

 if(idx<foundNtuplets->nbins()) {
   if (foundNtuplets->size(idx)>5) printf("ERROR %d, %d\n", idx, foundNtuplets->size(idx));
   assert(foundNtuplets->size(idx)<6);
   for (auto ih = foundNtuplets->begin(idx); ih!=foundNtuplets->end(idx); ++ih) assert(*ih<nHits);
 }
 #endif

 if (0==idx) {
   if (*nCells>=CAConstants::maxNumberOfDoublets()) printf("Cells overflow\n");
 }

 if (idx < (*nCells) ) {
   auto &thisCell = cells[idx];
   if (thisCell.theOuterNeighbors.full()) //++tooManyNeighbors[thisCell.theLayerPairId];
     printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
   if (thisCell.theTracks.full()) //++tooManyTracks[thisCell.theLayerPairId];
     printf("Tracks overflow %d in %d\n", idx, thisCell.theLayerPairId);
   if (thisCell.theDoubletId<0) atomicAdd(&killedCell,1);
 }
 if (idx < nHits) {
   if (isOuterHitOfCell[idx].full()) // ++tooManyOuterHitOfCell;
     printf("OuterHitOfCell overflow %d\n", idx);
 }

 __syncthreads();
// if (threadIdx.x==0) printf("number of killed cells %d\n",killedCell);
}


__global__
void
kernel_fishboneCleaner(GPUCACell const * cells, uint32_t const * __restrict__ nCells,
                            pixelTuplesHeterogeneousProduct::Quality * quality
                           ) {

  constexpr auto bad = pixelTuplesHeterogeneousProduct::bad;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells) ) return;
  auto const & thisCell = cells[cellIndex];
  if (thisCell.theDoubletId>=0) return;

  for (auto it : thisCell.theTracks) quality[it] = bad;

}

__global__
void
kernel_fastDuplicateRemover(GPUCACell const * cells, uint32_t const * __restrict__ nCells,
                            TuplesOnGPU::Container * foundNtuplets,
                            Rfit::helix_fit const * __restrict__ hfit,
                            pixelTuplesHeterogeneousProduct::Quality * quality
                           ) {

  constexpr auto bad = pixelTuplesHeterogeneousProduct::bad;
  constexpr auto dup = pixelTuplesHeterogeneousProduct::dup;
  // constexpr auto loose = pixelTuplesHeterogeneousProduct::loose;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (cellIndex >= (*nCells) ) return;
  auto const & thisCell = cells[cellIndex];
  if (thisCell.theDoubletId<0) return;

  float mc=1000.f; uint16_t im=60000; uint32_t maxNh=0;
   
  // find maxNh
  for (auto it : thisCell.theTracks) {
    if (quality[it] == bad) continue;
    auto nh = foundNtuplets->size(it);
    maxNh = std::max(nh,maxNh);
  }
  // find min chi2
  for (auto it : thisCell.theTracks) {
    auto nh = foundNtuplets->size(it);
    if (nh!=maxNh) continue; 
    if (quality[it]!= bad && 
        hfit[it].chi2_line+hfit[it].chi2_circle < mc) {
      mc=hfit[it].chi2_line+hfit[it].chi2_circle;
      im=it;
    }
  }
  // mark duplicates
  for (auto it : thisCell.theTracks) {
     if (quality[it]!= bad && it!=im) quality[it] = dup; //no race:  simple assignment of the same constant
  }
}

__global__ 
void
kernel_connect(AtomicPairCounter * apc1, AtomicPairCounter * apc2,  // just to zero them,
               GPUCACell::Hits const *  __restrict__ hhp,
               GPUCACell * cells, uint32_t const * __restrict__ nCells,
               GPUCACell::OuterHitOfCell const * __restrict__ isOuterHitOfCell) {

  auto const & hh = *hhp;

  // 87 cm/GeV = 1/(3.8T * 0.3)
  // take less than radius given by the hardPtCut and reject everything below
  // auto hardCurvCut = 1.f/(hardPtCut * 87.f);
  constexpr auto hardCurvCut = 1.f/(0.35f * 87.f); // FIXME VI tune
  constexpr auto ptmin = 0.9f; // FIXME original "tune"

  auto cellIndex = threadIdx.y + blockIdx.y * blockDim.y;
  auto first = threadIdx.x;
  auto stride = blockDim.x;

  if (0==(cellIndex+first)) { (*apc1)=0; (*apc2)=0; }// ready for next kernel

  if (cellIndex >= (*nCells) ) return;
  auto const & thisCell = cells[cellIndex];
  if (thisCell.theDoubletId<0) return;
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  auto vi = isOuterHitOfCell[innerHitId].data();
  for (auto j = first; j < numberOfPossibleNeighbors; j+=stride) {
     auto otherCell = __ldg(vi+j);
     if (cells[otherCell].theDoubletId<0) continue;
     if (thisCell.check_alignment(hh,
                 cells[otherCell], ptmin, hardCurvCut)
        ) {
          cells[otherCell].theOuterNeighbors.push_back(cellIndex);
     }
  }
}

__global__ 
void kernel_find_ntuplets(
    GPUCACell::Hits const *  __restrict__ hhp,
    GPUCACell * __restrict__ cells, uint32_t const * nCells,
    TuplesOnGPU::Container * foundNtuplets, AtomicPairCounter * apc,
    GPUCACell::TupleMultiplicity * tupleMultiplicity,
    unsigned int minHitsPerNtuplet)
{

  // recursive: not obvious to widen
  auto const & hh = *hhp;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];

#ifdef CA_USE_LOCAL_COUNTERS
  __shared__ GPUCACell::TupleMultiplicity::CountersOnly local;
  if (0==threadIdx.x) local.zero();
  __syncthreads();
#endif

  if (thisCell.theLayerPairId==0 || thisCell.theLayerPairId==3 || thisCell.theLayerPairId==8) { // inner layer is 0 FIXME
    GPUCACell::TmpTuple stack;
    stack.reset();
    thisCell.find_ntuplets(hh, cells, *foundNtuplets, *apc, 
                           #ifdef CA_USE_LOCAL_COUNTERS
                           local,
                           #else
                           *tupleMultiplicity,
                           #endif 
                           stack, minHitsPerNtuplet);
    assert(stack.size()==0);
    // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
  }

#ifdef CA_USE_LOCAL_COUNTERS
  __syncthreads(); 
  if (0==threadIdx.x) tupleMultiplicity->add(local);
#endif
}


__global__
void kernel_fillMultiplicity(
      TuplesOnGPU::Container const * __restrict__ foundNtuplets, 
      GPUCACell::TupleMultiplicity * tupleMultiplicity
     )
{
  auto it = blockIdx.x * blockDim.x + threadIdx.x;

  if (it>=foundNtuplets->nbins()) return;

  auto nhits = foundNtuplets->size(it);
  if (nhits<3) return;
  tupleMultiplicity->fillDirect(nhits,it);
}


__global__
void kernel_VerifyFit(TuplesOnGPU::Container const * __restrict__ tuples,
                 Rfit::helix_fit const *  __restrict__ fit_results,
                 Quality *  __restrict__ quality) {

  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>= tuples->nbins()) return;
  if (tuples->size(idx)==0) {
    return;
  }

  quality[idx] = pixelTuplesHeterogeneousProduct::bad;

  if (tuples->size(idx)<3) { 
    return;
  }

  bool isNaN = false;
  for (int i=0; i<5; ++i) {
    isNaN |=  fit_results[idx].par(i)!=fit_results[idx].par(i);
  }
  isNaN |=  !(fit_results[idx].chi2_line+fit_results[idx].chi2_circle < 100.f);  // catch NaN as well

#ifdef GPU_DEBUG
 if (isNaN) printf("NaN or Bad Fit %d size %d chi2 %f/%f\n",idx,tuples->size(idx), fit_results[idx].chi2_line,fit_results[idx].chi2_circle);
#endif

  // impose "region cuts" (NaN safe)
  // phi,Tip,pt,cotan(theta)),Zip
  bool ok = std::abs(fit_results[idx].par(1)) < ( tuples->size(idx)>3 ? 0.1f : 0.05f) 
         && fit_results[idx].par(2) > ( tuples->size(idx)>3 ? 0.3f : 0.5f)
         && std::abs(fit_results[idx].par(4)) < 12.f;
  
  ok &= (!isNaN);
  quality[idx] = ok ? pixelTuplesHeterogeneousProduct::loose : pixelTuplesHeterogeneousProduct::bad; 
}

__global__
void kernel_print_found_ntuplets(TuplesOnGPU::Container * foundNtuplets, uint32_t maxPrint) {
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->size()); ++i) {
    printf("\nquadruplet %d: %d %d %d %d\n", i,
           (*(*foundNtuplets).begin(i)),
           (*(*foundNtuplets).begin(i)+1),
           (*(*foundNtuplets).begin(i)+2),
           (*(*foundNtuplets).begin(i)+3)
          );
  }
}

void CAHitQuadrupletGeneratorKernels::launchKernels( // here goes algoparms....
                                                HitsOnCPU const & hh,
                                                TuplesOnGPU & tuples_d,
                                                cudaStream_t cudaStream)
{
  auto & gpu_ = tuples_d;
  auto maxNumberOfDoublets_ = CAConstants::maxNumberOfDoublets();


  auto nhits = hh.nHits;
  assert(nhits <= PixelGPUConstants::maxNumberOfHits);
  
  if (earlyFishbone_) {
    auto nthTot = 64;
    auto stride = 4;
    auto blockSize = nthTot/stride;
    auto numberOfBlocks = (nhits + blockSize - 1)/blockSize;
    dim3 blks(1,numberOfBlocks,1);
    dim3 thrs(stride,blockSize,1);
    fishbone<<<blks,thrs, 0, cudaStream>>>(
      hh.gpu_d,
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      nhits, false
    );
    cudaCheck(cudaGetLastError());
  }

  auto nthTot = 64;
  auto stride = 4;
  auto blockSize = nthTot/stride;
  auto numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1)/blockSize;
  dim3 blks(1,numberOfBlocks,1);
  dim3 thrs(stride,blockSize,1);

  kernel_connect<<<blks, thrs, 0, cudaStream>>>(
      gpu_.apc_d, device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
      hh.gpu_d,
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_
  );
  cudaCheck(cudaGetLastError());

  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      hh.gpu_d,
      device_theCells_, device_nCells_,
      gpu_.tuples_d,
      gpu_.apc_d,
      device_tupleMultiplicity_,
      minHitsPerNtuplet_      
  );
  cudaCheck(cudaGetLastError());

  numberOfBlocks = (TuplesOnGPU::Container::totbins() + blockSize - 1)/blockSize;
  cudautils::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(gpu_.apc_d,gpu_.tuples_d);

  cudautils::launchFinalize(device_tupleMultiplicity_,device_tmws_,cudaStream);


  blockSize = 128;
  numberOfBlocks = (CAConstants::maxTuples() + blockSize - 1) / blockSize;
  kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(gpu_.tuples_d,device_tupleMultiplicity_);
  cudaCheck(cudaGetLastError());

  if (lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot/stride;
    auto numberOfBlocks = (nhits + blockSize - 1)/blockSize;
    dim3 blks(1,numberOfBlocks,1);
    dim3 thrs(stride,blockSize,1);
    fishbone<<<blks,thrs, 0, cudaStream>>>(
      hh.gpu_d,
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      nhits, true
    );
    cudaCheck(cudaGetLastError());
  }

#ifndef NO_CHECK_OVERFLOWS
  numberOfBlocks = (std::max(nhits, maxNumberOfDoublets_) + blockSize - 1)/blockSize;
  kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
                        gpu_.tuples_d, gpu_.apc_d,
                        device_theCells_, device_nCells_,
                        device_isOuterHitOfCell_, nhits
                       );
  cudaCheck(cudaGetLastError());
#endif


  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(gpu_.tuples_d, 10);
  }


void CAHitQuadrupletGeneratorKernels::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
  auto nhits = hh.nHits;

  int stride=1;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize/stride;
  int blocks = (3 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1,blocks,1);
  dim3 thrs(stride,threadsPerBlock,1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0, stream>>>(
            device_theCells_, device_nCells_, hh.gpu_d, device_isOuterHitOfCell_);
  cudaCheck(cudaGetLastError());
}

void CAHitQuadrupletGeneratorKernels::classifyTuples(HitsOnCPU const & hh, TuplesOnGPU & tuples, cudaStream_t cudaStream) {
    auto blockSize = 64;
    auto numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1)/blockSize;
    kernel_VerifyFit<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples.tuples_d, tuples.helix_fit_results_d, tuples.quality_d);

    numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1)/blockSize;
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_theCells_, device_nCells_,tuples.quality_d);

    numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1)/blockSize;
    kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_theCells_, device_nCells_,tuples.tuples_d,tuples.helix_fit_results_d, tuples.quality_d);

}
