#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h

#include <Eigen/Core>

#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include "GPUCACell.h"

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
    Rfit::line_fit *line_fit);

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
    Rfit::line_fit *line_fit);

void wrapperLineFitAllHits(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> * foundNtuplets,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Eigen::Vector4d *fast_fit,
    Rfit::line_fit *line_fit);

void wrapperCheckOverflows(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    GPUCACell const * cells,
    uint32_t const * nCells,
    GPU::VecArray<unsigned int, 256> const * isOuterHitOfCell,
    uint32_t nHits,
    uint32_t maxNumberOfDoublets);

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
    unsigned int maxNumberOfHits);

void wrapperFindNtuplets(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPUCACell * const cells,
    uint32_t const * nCells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets);
  
void wrapperPrintFoundNtuplets(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    int maxPrint);

void wrapperDoubletsFromHisto(
    int gridSize, int blockSize, cudaStream_t cudaStream,
    GPUCACell * cells,
    uint32_t * nCells,
    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hits,
    GPU::VecArray<unsigned int, 256> * isOuterHitOfCell);

#endif // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorKernels_h
