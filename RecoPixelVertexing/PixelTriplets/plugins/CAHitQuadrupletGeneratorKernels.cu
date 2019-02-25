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
               uint32_t nHits, CAHitQuadrupletGeneratorKernels::Counters * counters) {

 __shared__ uint32_t killedCell;
 killedCell=0;
 __syncthreads();

 auto idx = threadIdx.x + blockIdx.x * blockDim.x;

 auto    & c = *counters;
 // counters once per event
 if(0==idx) {
   atomicAdd(&c.nEvents,1);
   atomicAdd(&c.nHits,nHits);
   atomicAdd(&c.nCells,*nCells);   
   atomicAdd(&c.nTuples,apc->get().m);
 }
  
#ifdef GPU_DEBUG
 if (0==idx) {
   printf("number of found cells %d, found tuples %d with total hits %d out of %d\n",*nCells, apc->get().m, apc->get().n, nHits);
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
   if (apc->get().m >=CAConstants::maxNumberOfQuadruplets()) printf("Tuples overflow\n");
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
 if (threadIdx.x==0) atomicAdd(&c.nKilledCells,killedCell);
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
   
  auto score = [&](auto it) {
    return std::abs(hfit[it].par(1));  // tip
    // return hfit[it].chi2_line+hfit[it].chi2_circle;  //chi2 
  };

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
        score(it) < mc) {
      mc=score(it);
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
void kernel_classifyTracks(TuplesOnGPU::Container const * __restrict__ tuples,
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

  const float par[4] = {0.68177776,  0.74609577, -0.08035491,  0.00315399};
  constexpr float chi2CutFact = 30;  // *0.7 if Riemann....
  auto chi2Cut = [&](float pt) {
     pt = std::min(pt,10.f);
     return chi2CutFact*(par[0]+pt*(par[1]+pt*(par[2]+pt*par[3]))); 
  };

  bool isNaN = false;
  for (int i=0; i<5; ++i) {
    isNaN |=  fit_results[idx].par(i)!=fit_results[idx].par(i);
  }
  isNaN |=  !(fit_results[idx].chi2_line+fit_results[idx].chi2_circle < chi2Cut(fit_results[idx].par(2)));  // catch NaN as well

#ifdef GPU_DEBUG
 if (isNaN) printf("NaN or Bad Fit %d size %d chi2 %f/%f\n",idx,tuples->size(idx), fit_results[idx].chi2_line,fit_results[idx].chi2_circle);
#endif

  // impose "region cuts" (NaN safe)
  // phi,Tip,pt,cotan(theta)),Zip
  bool ok = std::abs(fit_results[idx].par(1)) < ( tuples->size(idx)>3 ? 0.5f : 0.3f) 
         && fit_results[idx].par(2) > ( tuples->size(idx)>3 ? 0.3f : 0.5f)
         && std::abs(fit_results[idx].par(4)) < 12.f;
  
  ok &= (!isNaN);
  quality[idx] = ok ? pixelTuplesHeterogeneousProduct::loose : pixelTuplesHeterogeneousProduct::bad; 
}

__global__
void kernel_doStatsForTracks(TuplesOnGPU::Container const * __restrict__ tuples,
                        Quality const *  __restrict__ quality,
                        CAHitQuadrupletGeneratorKernels::Counters * counters) {

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x*blockDim.x) {
    if (tuples->size(idx)==0) continue;
    if(quality[idx] != pixelTuplesHeterogeneousProduct::loose ) continue;
    atomicAdd(&(counters->nGoodTracks),1);
  }
}

__global__
void kernel_countHitInTracks(TuplesOnGPU::Container const * __restrict__ tuples,
                            Quality const *  __restrict__ quality,
                            CAHitQuadrupletGeneratorKernels::HitToTuple * hitToTuple) {

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x*blockDim.x) {
    if (tuples->size(idx)==0) continue;
    if(quality[idx] != pixelTuplesHeterogeneousProduct::loose ) continue;
    for (auto h = tuples->begin(idx); h!= tuples->end(idx); ++h)
      hitToTuple->countDirect(*h);
  }
}

__global__
void kernel_fillHitInTracks(TuplesOnGPU::Container const * __restrict__ tuples,
                                  Quality const *  __restrict__ quality,
                           CAHitQuadrupletGeneratorKernels::HitToTuple * hitToTuple) {

  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = tuples->nbins(); idx < ntot; idx += gridDim.x*blockDim.x) {
    if (tuples->size(idx)==0) continue;
    if(quality[idx] != pixelTuplesHeterogeneousProduct::loose ) continue;
    for (auto h = tuples->begin(idx); h!= tuples->end(idx); ++h)
      hitToTuple->fillDirect(*h,idx);
  }
}

__global__
void kernel_doStatsForHitInTracks(CAHitQuadrupletGeneratorKernels::HitToTuple const * __restrict__ hitToTuple,
                                  CAHitQuadrupletGeneratorKernels::Counters * counters) {
  auto    & c = *counters;
  int first = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = first, ntot = hitToTuple->nbins(); idx < ntot; idx += gridDim.x*blockDim.x) {
     if (hitToTuple->size(idx)==0) continue;
     atomicAdd(&c.nUsedHits,1);
     if (hitToTuple->size(idx)>1) atomicAdd(&c.nDupHits,1);
  }
}

__global__
void kernel_tripletCleaner(siPixelRecHitsHeterogeneousProduct::HitsOnGPU const *  __restrict__ hhp,
                           TuplesOnGPU::Container const * __restrict__ ptuples,
                           Rfit::helix_fit const * __restrict__ hfit,
                           Quality *  __restrict__ quality,
                           CAHitQuadrupletGeneratorKernels::HitToTuple const * __restrict__ phitToTuple
                          ) {

  constexpr auto bad = pixelTuplesHeterogeneousProduct::bad;
  constexpr auto dup = pixelTuplesHeterogeneousProduct::dup;
  // constexpr auto loose = pixelTuplesHeterogeneousProduct::loose;

  auto & hitToTuple = *phitToTuple;
  auto const & foundNtuplets = *ptuples;

  //  auto const & hh = *hhp;
  // auto l1end = hh.hitsLayerStart_d[1]; 

  int first = blockDim.x * blockIdx.x + threadIdx.x;

  for (int idx = first, ntot = hitToTuple.nbins(); idx < ntot; idx += gridDim.x*blockDim.x) {
     if (hitToTuple.size(idx)<2) continue;

     float mc=10000.f; uint16_t im=60000; 
     uint32_t maxNh=0;

     // find maxNh
     for (auto it=hitToTuple.begin(idx); it!=hitToTuple.end(idx); ++it) {
       uint32_t nh = foundNtuplets.size(*it);
       maxNh = std::max(nh,maxNh);
     }
     // kill all tracks shorter than maxHn (only triplets???)
     for (auto it=hitToTuple.begin(idx); it!=hitToTuple.end(idx); ++it) {
       uint32_t nh = foundNtuplets.size(*it);
       if (maxNh!=nh) quality[*it] = dup;
     }
  
     if (maxNh>3) continue;
     // if (idx>=l1end) continue;  // only for layer 1
     // for triplets choose best tip!
     for (auto ip=hitToTuple.begin(idx); ip!=hitToTuple.end(idx); ++ip) {
       auto const it = *ip;
       if (quality[it]!= bad &&
         std::abs(hfit[it].par(1)) < mc) {
         mc=std::abs(hfit[it].par(1));
         im=it;
       }
     }
     // mark duplicates
     for (auto ip=hitToTuple.begin(idx); ip!=hitToTuple.end(idx); ++ip) {
       auto const it = *ip;
       if (quality[it]!= bad && it!=im  
          ) quality[it] = dup; //no race:  simple assignment of the same constant
     }
  }  // loop over hits


} 


__global__
void kernel_print_found_ntuplets(TuplesOnGPU::Container const * __restrict__ foundNtuplets, uint32_t maxPrint) {
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->nbins()); ++i) {
    if (foundNtuplets->size(i)<4) continue;
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
  auto rescale = numberOfBlocks/65536;
  blockSize*=(rescale+1);
  numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1)/blockSize;
  assert(numberOfBlocks<65536);
  assert(blockSize>0 && 0==blockSize%16);
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

  if (doStats_) {
    numberOfBlocks = (std::max(nhits, maxNumberOfDoublets_) + blockSize - 1)/blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
                        gpu_.tuples_d, gpu_.apc_d,
                        device_theCells_, device_nCells_,
                        device_isOuterHitOfCell_, nhits,
                        counters_
                       );
    cudaCheck(cudaGetLastError());
  }


  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(gpu_.tuples_d, 10);
  }


void CAHitQuadrupletGeneratorKernels::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
  auto nhits = hh.nHits;

  int stride=1;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize/stride;
  int blocks = (2 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1,blocks,1);
  dim3 thrs(stride,threadsPerBlock,1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0, stream>>>(
            device_theCells_, device_nCells_, hh.gpu_d, device_isOuterHitOfCell_, idealConditions_);
  cudaCheck(cudaGetLastError());
}

void CAHitQuadrupletGeneratorKernels::classifyTuples(HitsOnCPU const & hh, TuplesOnGPU & tuples, cudaStream_t cudaStream) {
    auto blockSize = 64;

    // classify tracks based on kinematics
    auto numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1)/blockSize;
    kernel_classifyTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples.tuples_d, tuples.helix_fit_results_d, tuples.quality_d);

    // apply fishbone cleaning to good tracks
    numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1)/blockSize;
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_theCells_, device_nCells_,tuples.quality_d);

    // remove duplicates (tracks that share a doublet) 
    numberOfBlocks = (CAConstants::maxNumberOfDoublets() + blockSize - 1)/blockSize;
    kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_theCells_, device_nCells_,tuples.tuples_d,tuples.helix_fit_results_d, tuples.quality_d);

    // fill hit->track "map"
    numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1)/blockSize;
    kernel_countHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples.tuples_d,tuples.quality_d,device_hitToTuple_);
    cudautils::launchFinalize(device_hitToTuple_,device_tmws_,cudaStream);
    kernel_fillHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples.tuples_d,tuples.quality_d,device_hitToTuple_);

    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1)/blockSize;
    kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.gpu_d,tuples.tuples_d,tuples.helix_fit_results_d,tuples.quality_d,device_hitToTuple_);

    if (doStats_) {
      // counters (add flag???)
      numberOfBlocks = (HitToTuple::capacity() + blockSize - 1)/blockSize;
      kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitToTuple_, counters_);
      numberOfBlocks = (CAConstants::maxNumberOfQuadruplets() + blockSize - 1)/blockSize;
      kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples.tuples_d,tuples.quality_d,counters_);
    }
}


__global__
void kernel_printCounters(CAHitQuadrupletGeneratorKernels::Counters const * counters) {
   
   auto const & c = *counters;
   printf("Counters Raw %lld %lld %lld %lld %lld %lld %lld %lld\n",c.nEvents,c.nHits,c.nCells,c.nTuples,c.nGoodTracks,c.nUsedHits, c.nDupHits, c.nKilledCells);
   printf("Counters Norm %lld ||  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f|  %.1f||\n",c.nEvents,c.nHits/double(c.nEvents),c.nCells/double(c.nEvents),
                                                c.nTuples/double(c.nEvents),c.nGoodTracks/double(c.nEvents),
                                                c.nUsedHits/double(c.nEvents),c.nDupHits/double(c.nEvents),c.nKilledCells/double(c.nEvents));

}

void CAHitQuadrupletGeneratorKernels::printCounters() const {
  kernel_printCounters<<<1,1>>>(counters_);
}
