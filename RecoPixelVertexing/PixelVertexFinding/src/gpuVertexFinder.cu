#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksByDensity.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksDBSCAN.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksIterative.h"

#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

// a macro SORRY
#define LOC_ZV(M) ((char*)(gpu_d) + offsetof(ZVertices, M))
#define LOC_WS(M) ((char*)(ws_d) + offsetof(WorkSpace, M))

namespace gpuVertexFinder {

  void Producer::allocate() {
    cudaCheck(cudaMalloc(&gpu_d, sizeof(ZVertices)));
    cudaCheck(cudaMalloc(&ws_d, sizeof(WorkSpace)));
  }

  void Producer::deallocate() {
    cudaCheck(cudaFree(gpu_d));
    cudaCheck(cudaFree(ws_d));
  }

  __global__ void loadTracks(pixelTuplesHeterogeneousProduct::TuplesOnGPU const* tracks, WorkSpace* pws, float ptMin) {
    auto const& tuples = *tracks->tuples_d;
    auto const* fit = tracks->helix_fit_results_d;
    auto const* quality = tracks->quality_d;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tuples.nbins())
      return;
    if (tuples.size(idx) == 0) {
      return;
    }

    if (tuples.size(idx) < 4)
      return;  // no triplets
    if (quality[idx] != pixelTuplesHeterogeneousProduct::loose)
      return;

    auto const& fittedTrack = fit[idx];

    if (fittedTrack.par(2) < ptMin)
      return;

    auto& data = *pws;
    auto it = atomicAdd(&data.ntrks, 1);
    data.itrk[it] = idx;
    data.zt[it] = fittedTrack.par(4);
    data.ezt2[it] = fittedTrack.cov(4, 4);
    data.ptt2[it] = fittedTrack.par(2) * fittedTrack.par(2);
  }

  void Producer::produce(cudaStream_t stream, TuplesOnCPU const& tracks, float ptMin) {
    assert(gpu_d);
    assert(ws_d);
    assert(tracks.gpu_d);
    init<<<1, 1, 0, stream>>>(gpu_d, ws_d);
    auto blockSize = 128;
    auto numberOfBlocks = (CAConstants::maxTuples() + blockSize - 1) / blockSize;
    loadTracks<<<numberOfBlocks, blockSize, 0, stream>>>(tracks.gpu_d, ws_d, ptMin);
    cudaCheck(cudaGetLastError());
    if (useDensity_) {
      clusterTracksByDensity<<<1, 1024 - 256, 0, stream>>>(gpu_d, ws_d, minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN<<<1, 1024 - 256, 0, stream>>>(gpu_d, ws_d, minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative<<<1, 1024 - 256, 0, stream>>>(gpu_d, ws_d, minT, eps, errmax, chi2max);
    }
    cudaCheck(cudaGetLastError());
    fitVertices<<<1, 1024 - 256, 0, stream>>>(gpu_d, ws_d, 50.);
    cudaCheck(cudaGetLastError());

    splitVertices<<<1024, 128, 0, stream>>>(gpu_d, ws_d, 9.f);
    cudaCheck(cudaGetLastError());
    fitVertices<<<1, 1024 - 256, 0, stream>>>(gpu_d, ws_d, 5000.);
    cudaCheck(cudaGetLastError());

    sortByPt2<<<1, 256, 0, stream>>>(gpu_d, ws_d);
    cudaCheck(cudaGetLastError());

    if (enableTransfer) {
      auto from = (char*)(gpu_d) + offsetof(ZVertices, nvFinal);
      cudaCheck(cudaMemcpyAsync(&gpuProduct.nVertices, from, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
      from = (char*)(ws_d) + offsetof(WorkSpace, ntrks);
      cudaCheck(cudaMemcpyAsync(&gpuProduct.nTracks, from, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    }
  }

  Producer::OnCPU const& Producer::fillResults(cudaStream_t stream) {
    if (!enableTransfer)
      return gpuProduct;

    // finish copy
    // FIXME copy to zv idv
    gpuProduct.ivtx.resize(gpuProduct.nTracks);
    cudaCheck(cudaMemcpyAsync(
        gpuProduct.ivtx.data(), LOC_WS(iv), sizeof(int32_t) * gpuProduct.nTracks, cudaMemcpyDeviceToHost, stream));
    gpuProduct.itrk.resize(gpuProduct.nTracks);
    cudaCheck(cudaMemcpyAsync(
        gpuProduct.itrk.data(), LOC_WS(itrk), sizeof(int16_t) * gpuProduct.nTracks, cudaMemcpyDeviceToHost, stream));

    gpuProduct.z.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(
        gpuProduct.z.data(), LOC_ZV(zv), sizeof(float) * gpuProduct.nVertices, cudaMemcpyDeviceToHost, stream));
    gpuProduct.zerr.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(
        gpuProduct.zerr.data(), LOC_ZV(wv), sizeof(float) * gpuProduct.nVertices, cudaMemcpyDeviceToHost, stream));
    gpuProduct.chi2.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(
        gpuProduct.chi2.data(), LOC_ZV(chi2), sizeof(float) * gpuProduct.nVertices, cudaMemcpyDeviceToHost, stream));

    gpuProduct.sortInd.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.sortInd.data(),
                              LOC_ZV(sortInd),
                              sizeof(uint16_t) * gpuProduct.nVertices,
                              cudaMemcpyDeviceToHost,
                              stream));

    cudaStreamSynchronize(stream);

    return gpuProduct;
  }

}  // namespace gpuVertexFinder

#undef FROM
