#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

#undef PIXVERTEX_DEBUG_PRODUCE

namespace gpuVertexFinder {

  // reject outlier tracks that contribute more than this to the chi2 of the vertex fit
  constexpr float maxChi2ForFirstFit = 50.f;
  constexpr float maxChi2ForFinalFit = 5000.f;

  // split vertices with a chi2/NDoF greater than this
  constexpr float maxChi2ForSplit = 9.f;

  __global__ void loadTracks(TkSoA const* ptracks, ZVertexSoA* soa, WorkSpace* pws, float ptMin) {
    assert(ptracks);
    assert(soa);
    auto const& tracks = *ptracks;
    auto const& fit = tracks.stateAtBS;
    auto const* quality = tracks.qualityData();

    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = first, nt = TkSoA::stride(); idx < nt; idx += gridDim.x * blockDim.x) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0)
        break;  // this is a guard: maybe we need to move to nTracks...

      // initialize soa...
      soa->idv[idx] = -1;

      if (nHits < 4)
        continue;  // no triplets
      if (quality[idx] < pixelTrack::Quality::highPurity)
        continue;

      auto pt = tracks.pt(idx);

      if (pt < ptMin)
        continue;

      auto& data = *pws;
      auto it = atomicAdd(&data.ntrks, 1);
      data.itrk[it] = idx;
      data.zt[it] = tracks.zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt * pt;
    }
  }

// #define THREE_KERNELS
#ifndef THREE_KERNELS
  __global__ void vertexFinderOneKernel(gpuVertexFinder::ZVertices* pdata,
                                        gpuVertexFinder::WorkSpace* pws,
                                        int minT,      // min number of neighbours to be "seed"
                                        float eps,     // max absolute distance to cluster
                                        float errmax,  // max error to be "seed"
                                        float chi2max  // max normalized distance to cluster,
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
    __syncthreads();
    fitVertices(pdata, pws, maxChi2ForFirstFit);
    __syncthreads();
    splitVertices(pdata, pws, maxChi2ForSplit);
    __syncthreads();
    fitVertices(pdata, pws, maxChi2ForFinalFit);
    __syncthreads();
    sortByPt2(pdata, pws);
  }
#else
  __global__ void vertexFinderKernel1(gpuVertexFinder::ZVertices* pdata,
                                      gpuVertexFinder::WorkSpace* pws,
                                      int minT,      // min number of neighbours to be "seed"
                                      float eps,     // max absolute distance to cluster
                                      float errmax,  // max error to be "seed"
                                      float chi2max  // max normalized distance to cluster,
  ) {
    clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
    __syncthreads();
    fitVertices(pdata, pws, maxChi2ForFirstFit);
  }

  __global__ void vertexFinderKernel2(gpuVertexFinder::ZVertices* pdata, gpuVertexFinder::WorkSpace* pws) {
    fitVertices(pdata, pws, maxChi2ForFinalFit);
    __syncthreads();
    sortByPt2(pdata, pws);
  }
#endif

#ifdef __CUDACC__
  ZVertexHeterogeneous Producer::makeAsync(cudaStream_t stream, TkSoA const* tksoa, float ptMin) const {
#ifdef PIXVERTEX_DEBUG_PRODUCE
    std::cout << "producing Vertices on GPU" << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
    ZVertexHeterogeneous vertices(cms::cuda::make_device_unique<ZVertexSoA>(stream));
#else
  ZVertexHeterogeneous Producer::make(TkSoA const* tksoa, float ptMin) const {
#ifdef PIXVERTEX_DEBUG_PRODUCE
    std::cout << "producing Vertices on  CPU" << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
    ZVertexHeterogeneous vertices(std::make_unique<ZVertexSoA>());
#endif
    assert(tksoa);
    auto* soa = vertices.get();
    assert(soa);

#ifdef __CUDACC__
    auto ws_d = cms::cuda::make_device_unique<WorkSpace>(stream);
#else
    auto ws_d = std::make_unique<WorkSpace>();
#endif

#ifdef __CUDACC__
    init<<<1, 1, 0, stream>>>(soa, ws_d.get());
    auto blockSize = 128;
    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    loadTracks<<<numberOfBlocks, blockSize, 0, stream>>>(tksoa, soa, ws_d.get(), ptMin);
    cudaCheck(cudaGetLastError());
#else
    init(soa, ws_d.get());
    loadTracks(tksoa, soa, ws_d.get(), ptMin);
#endif

#ifdef __CUDACC__
    // Running too many thread lead to problems when printf is enabled.
    constexpr int maxThreadsForPrint = 1024 - 128;
    constexpr int numBlocks = 1024;
    constexpr int threadsPerBlock = 128;

    if (oneKernel_) {
      // implemented only for density clustesrs
#ifndef THREE_KERNELS
      vertexFinderOneKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
#else
      vertexFinderKernel1<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      cudaCheck(cudaGetLastError());
      // one block per vertex...
      splitVerticesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(soa, ws_d.get(), maxChi2ForSplit);
      cudaCheck(cudaGetLastError());
      vertexFinderKernel2<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get());
#endif
    } else {  // five kernels
      if (useDensity_) {
        clusterTracksByDensityKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      } else if (useDBSCAN_) {
        clusterTracksDBSCAN<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      } else if (useIterative_) {
        clusterTracksIterative<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      }
      cudaCheck(cudaGetLastError());
      fitVerticesKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), maxChi2ForFirstFit);
      cudaCheck(cudaGetLastError());
      // one block per vertex...
      splitVerticesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(soa, ws_d.get(), maxChi2ForSplit);
      cudaCheck(cudaGetLastError());
      fitVerticesKernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get(), maxChi2ForFinalFit);
      cudaCheck(cudaGetLastError());
      sortByPt2Kernel<<<1, maxThreadsForPrint, 0, stream>>>(soa, ws_d.get());
    }
    cudaCheck(cudaGetLastError());
#else  // __CUDACC__
    if (useDensity_) {
      clusterTracksByDensity(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative(soa, ws_d.get(), minT, eps, errmax, chi2max);
    }
#ifdef PIXVERTEX_DEBUG_PRODUCE
    std::cout << "found " << (*ws_d).nvIntermediate << " vertices " << std::endl;
#endif  // PIXVERTEX_DEBUG_PRODUCE
    fitVertices(soa, ws_d.get(), maxChi2ForFirstFit);
    // one block per vertex!
    splitVertices(soa, ws_d.get(), maxChi2ForSplit);
    fitVertices(soa, ws_d.get(), maxChi2ForFinalFit);
    sortByPt2(soa, ws_d.get());
#endif

    return vertices;
  }

}  // namespace gpuVertexFinder
