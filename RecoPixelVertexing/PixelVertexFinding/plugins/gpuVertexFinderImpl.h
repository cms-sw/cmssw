#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "gpuClusterTracksByDensity.h"
#include "gpuClusterTracksDBSCAN.h"
#include "gpuClusterTracksIterative.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

namespace gpuVertexFinder {

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
      if (quality[idx] != trackQuality::loose)
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
    fitVertices(pdata, pws, 50.);
    __syncthreads();
    splitVertices(pdata, pws, 9.f);
    __syncthreads();
    fitVertices(pdata, pws, 5000.);
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
    fitVertices(pdata, pws, 50.);
  }

  __global__ void vertexFinderKernel2(gpuVertexFinder::ZVertices* pdata, gpuVertexFinder::WorkSpace* pws) {
    fitVertices(pdata, pws, 5000.);
    __syncthreads();
    sortByPt2(pdata, pws);
  }
#endif

#ifdef __CUDACC__
  ZVertexHeterogeneous Producer::makeAsync(cudaStream_t stream, TkSoA const* tksoa, float ptMin) const {
    // std::cout << "producing Vertices on GPU" << std::endl;
    ZVertexHeterogeneous vertices(cms::cuda::make_device_unique<ZVertexSoA>(stream));
#else
  ZVertexHeterogeneous Producer::make(TkSoA const* tksoa, float ptMin) const {
    // std::cout << "producing Vertices on  CPU" <<    std::endl;
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
    if (oneKernel_) {
      // implemented only for density clustesrs
#ifndef THREE_KERNELS
      vertexFinderOneKernel<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
#else
      vertexFinderKernel1<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      cudaCheck(cudaGetLastError());
      // one block per vertex...
      splitVerticesKernel<<<1024, 128, 0, stream>>>(soa, ws_d.get(), 9.f);
      cudaCheck(cudaGetLastError());
      vertexFinderKernel2<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get());
#endif
    } else {  // five kernels
      if (useDensity_) {
        clusterTracksByDensityKernel<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      } else if (useDBSCAN_) {
        clusterTracksDBSCAN<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      } else if (useIterative_) {
        clusterTracksIterative<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
      }
      cudaCheck(cudaGetLastError());
      fitVerticesKernel<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), 50.);
      cudaCheck(cudaGetLastError());
      // one block per vertex...
      splitVerticesKernel<<<1024, 128, 0, stream>>>(soa, ws_d.get(), 9.f);
      cudaCheck(cudaGetLastError());
      fitVerticesKernel<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get(), 5000.);
      cudaCheck(cudaGetLastError());
      sortByPt2Kernel<<<1, 1024 - 256, 0, stream>>>(soa, ws_d.get());
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
    // std::cout << "found " << (*ws_d).nvIntermediate << " vertices " << std::endl;
    fitVertices(soa, ws_d.get(), 50.);
    // one block per vertex!
    splitVertices(soa, ws_d.get(), 9.f);
    fitVertices(soa, ws_d.get(), 5000.);
    sortByPt2(soa, ws_d.get());
#endif

    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
