#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include <Eigen/Core>

#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "CAHitQuadrupletGeneratorKernels.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"

__global__
void kernelFastFitAllHits(
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
      blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif

  hits[helix_start].resize(3, hits_in_fit);
  hits_cov[helix_start].resize(3 * hits_in_fit, 3 * hits_in_fit);

  // Prepare data structure
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    auto hit = (*foundNtuplets)[helix_start].hitId[i];
    //  printf("Hit global_x: %f\n", hhp->xg_d[hit]);
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
    //  printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hhp->detInd_d[hit],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);

    hits[helix_start].col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];

    for (auto j = 0; j < 3; ++j) {
      for (auto l = 0; l < 3; ++l) {
        // Index numerology:
        // i: index of the hits/point (0,..,3)
        // j: index of space component (x,y,z)
        // l: index of space components (x,y,z)
        // ge is always in sync with the index i and is formatted as:
        // ge[] ==> [xx, xy, xz, yy, yz, zz]
        // in (j,l) notation, we have:
        // ge[] ==> [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
        // so the index ge_idx corresponds to the matrix elements:
        // | 0  1  2 |
        // | 1  3  4 |
        // | 2  4  5 |
        auto ge_idx = j + l + (j > 0 and l > 0);
        hits_cov[helix_start](i + j * hits_in_fit, i + l * hits_in_fit) = ge[ge_idx];
      }
    }
  }
  fast_fit[helix_start] = Rfit::Fast_fit(hits[helix_start]);
}

void wrapperFastFitAllHits(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  kernelFastFitAllHits<<<gridSize, blockSize, 0, cudaStream>>>(
    foundNtuplets,
    hhp,
    hits_in_fit,
    B,
    results,
    hits,
    hits_cov,
    circle_fit,
    fast_fit,
    line_fit);
  cudaCheck(cudaGetLastError());
}

__global__
void kernelCircleFitAllHits(
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
         blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif
  auto n = hits[helix_start].cols();

  Rfit::VectorNd rad = (hits[helix_start].block(0, 0, 2, n).colwise().norm());

  circle_fit[helix_start] =
      Rfit::Circle_fit(hits[helix_start].block(0, 0, 2, n),
                       hits_cov[helix_start].block(0, 0, 2 * n, 2 * n),
                       fast_fit[helix_start], rad, B, true);

#ifdef GPU_DEBUG
  printf("kernelCircleFitAllHits circle.par(0): %d %f\n", helix_start, circle_fit[helix_start].par(0));
  printf("kernelCircleFitAllHits circle.par(1): %d %f\n", helix_start, circle_fit[helix_start].par(1));
  printf("kernelCircleFitAllHits circle.par(2): %d %f\n", helix_start, circle_fit[helix_start].par(2));
#endif
}

void wrapperCircleFitAllHits(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  kernelCircleFitAllHits<<<gridSize, blockSize, 0, cudaStream>>>(
    foundNtuplets,
    hits_in_fit,
    B,
    results,
    hits,
    hits_cov,
    circle_fit,
    fast_fit,
    line_fit);
  cudaCheck(cudaGetLastError());
}

__global__
void kernelLineFitAllHits(
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
         blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif

  line_fit[helix_start] = Rfit::Line_fit(hits[helix_start], hits_cov[helix_start], circle_fit[helix_start], fast_fit[helix_start], B, true);

  par_uvrtopak(circle_fit[helix_start], B, true);

  // Grab helix_fit from the proper location in the output vector
  auto & helix = results[helix_start];
  helix.par << circle_fit[helix_start].par, line_fit[helix_start].par;

  // TODO: pass properly error booleans

  helix.cov = Eigen::MatrixXd::Zero(5, 5);
  helix.cov.block(0, 0, 3, 3) = circle_fit[helix_start].cov;
  helix.cov.block(3, 3, 2, 2) = line_fit[helix_start].cov;

  helix.q = circle_fit[helix_start].q;
  helix.chi2_circle = circle_fit[helix_start].chi2;
  helix.chi2_line = line_fit[helix_start].chi2;

#ifdef GPU_DEBUG
  printf("kernelLineFitAllHits line.par(0): %d %f\n", helix_start, circle_fit[helix_start].par(0));
  printf("kernelLineFitAllHits line.par(1): %d %f\n", helix_start, line_fit[helix_start].par(1));
#endif
}

void wrapperLineFitAllHits(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  kernelLineFitAllHits<<<gridSize, blockSize, 0, cudaStream>>>(
    foundNtuplets,
    B,
    results,
    hits,
    hits_cov,
    circle_fit,
    fast_fit,
    line_fit);
  cudaCheck(cudaGetLastError());
}

__global__
void kernelCheckOverflows(
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    GPUCACell const * __restrict__ cells,
    uint32_t const * __restrict__ nCells,
    GPU::VecArray<unsigned int, 256> const * __restrict__ isOuterHitOfCell,
    uint32_t nHits,
    uint32_t maxNumberOfDoublets)
{
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

void wrapperCheckOverflows(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    GPUCACell const * cells,
    uint32_t const * nCells,
    GPU::VecArray<unsigned int, 256> const * isOuterHitOfCell,
    uint32_t nHits,
    uint32_t maxNumberOfDoublets)
{
  kernelCheckOverflows<<<gridSize, blockSize, 0, cudaStream>>>(
    foundNtuplets,
    cells,
    nCells,
    isOuterHitOfCell,
    nHits,
    maxNumberOfDoublets);
  cudaCheck(cudaGetLastError());
}

__global__ 
void kernelConnect(
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    GPUCACell::Hits const *  __restrict__ hhp,
    GPUCACell * cells, uint32_t const * __restrict__ nCells,
    GPU::VecArray<unsigned int, 256> const * __restrict__ isOuterHitOfCell,
    float ptmin,
    float region_origin_radius, const float thetaCut,
    const float phiCut, const float hardPtCut,
    unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_)
{
  auto const & hh = *hhp;

  constexpr float region_origin_x = 0.;
  constexpr float region_origin_y = 0.;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (0==cellIndex) foundNtuplets->reset(); // ready for next kernel

  if (cellIndex >= (*nCells) ) return;
  auto const & thisCell = cells[cellIndex];
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  auto vi = isOuterHitOfCell[innerHitId].data();
  for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
     auto otherCell = __ldg(vi+j);

     if (thisCell.check_alignment(hh,
                 cells[otherCell], ptmin, region_origin_x, region_origin_y,
                  region_origin_radius, thetaCut, phiCut, hardPtCut)
        ) {
          cells[otherCell].theOuterNeighbors.push_back(cellIndex);
     }
  }
}

void wrapperConnect(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    GPUCACell::Hits const * hhp,
    GPUCACell * cells,
    uint32_t const * nCells,
    GPU::VecArray<unsigned int, 256> const * isOuterHitOfCell,
    float ptmin,
    float region_origin_radius,
    const float thetaCut,
    const float phiCut,
    const float hardPtCut,
    unsigned int maxNumberOfDoublets_,
    unsigned int maxNumberOfHits_)
{
  kernelConnect<<<gridSize, blockSize, 0, cudaStream>>>(
    foundNtuplets,
    hhp,
    cells,
    nCells,
    isOuterHitOfCell,
    ptmin,
    region_origin_radius,
    thetaCut,
    phiCut,
    hardPtCut,
    maxNumberOfDoublets_,
    maxNumberOfHits_);
  cudaCheck(cudaGetLastError());
}

__global__
void kernelFindNtuplets(
    GPUCACell * const __restrict__ cells,
    uint32_t const * nCells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_)
{
  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells))
    return;
  auto & thisCell = cells[cellIndex];
  if (thisCell.theLayerPairId != 0 and thisCell.theLayerPairId != 3 and thisCell.theLayerPairId != 8)
    return; // inner layer is 0 FIXME
  GPU::VecArray<siPixelRecHitsHeterogeneousProduct::hindex_type, 3> stack;
  stack.reset();
  thisCell.find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);
  assert(stack.size()==0);
  // printf("in %d found quadruplets: %d\n", cellIndex, foundNtuplets->size());
}

void wrapperFindNtuplets(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPUCACell * const cells,
    uint32_t const * nCells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets)
{
  kernelFindNtuplets<<<gridSize, blockSize, 0, cudaStream>>>(
    cells,
    nCells,
    foundNtuplets,
    minHitsPerNtuplet,
    maxNumberOfDoublets);
  cudaCheck(cudaGetLastError());
}

__global__
void kernelPrintFoundNtuplets(GPU::SimpleVector<Quadruplet> *foundNtuplets, int maxPrint)
{
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->size()); ++i) {
    printf("\nquadruplet %d: %d %d %d %d\n", i,
           (*foundNtuplets)[i].hitId[0],
           (*foundNtuplets)[i].hitId[1],
           (*foundNtuplets)[i].hitId[2],
           (*foundNtuplets)[i].hitId[3]
          );

  }
}

void wrapperPrintFoundNtuplets(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    int maxPrint)
{
  kernelPrintFoundNtuplets<<<gridSize, blockSize, 0, cudaStream>>>(
    foundNtuplets,
    maxPrint);
  cudaCheck(cudaGetLastError());
}

void wrapperDoubletsFromHisto(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPUCACell * cells,
    uint32_t * nCells,
    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hits,
    GPU::VecArray<unsigned int, 256> * isOuterHitOfCell)
{
  gpuPixelDoublets::getDoubletsFromHisto<<<gridSize, blockSize, 0, cudaStream>>>(cells, nCells, hits, isOuterHitOfCell);
  cudaCheck(cudaGetLastError());
}
