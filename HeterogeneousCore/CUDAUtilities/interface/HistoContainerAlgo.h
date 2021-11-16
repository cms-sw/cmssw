#ifndef HeterogeneousCore_CUDAUtilities_interface_HistoContainerAlgo_h
#define HeterogeneousCore_CUDAUtilities_interface_HistoContainerAlgo_h

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#ifdef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/maxCoopBlocks.h"
#endif

namespace cms {
  namespace cuda {

    template <template <CountOrFill> typename Func, typename Histo, typename... Args>
    __global__ void kernel_populate(typename Histo::View view, typename Histo::View::Counter *ws, Args... args) {
      namespace cg = cooperative_groups;
      auto grid = cg::this_grid();
      auto histo = static_cast<Histo *>(view.assoc);
      zeroAndInitCoop(view);
      grid.sync();
      Func<CountOrFill::count>::countOrFill(histo, std::forward<Args>(args)...);
      grid.sync();
      finalizeCoop(view, ws);
      grid.sync();
      Func<CountOrFill::fill>::countOrFill(histo, std::forward<Args>(args)...);
    }

    template <typename Histo, typename T, CountOrFill cof>
    __device__ __inline__ void countOrFillFromVector(Histo *__restrict__ h,
                                                     uint32_t nh,
                                                     T const *__restrict__ v,
                                                     uint32_t const *__restrict__ offsets) {
      int first = blockDim.x * blockIdx.x + threadIdx.x;
      for (int i = first, nt = offsets[nh]; i < nt; i += gridDim.x * blockDim.x) {
        auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
        assert((*off) > 0);
        int32_t ih = off - offsets - 1;
        assert(ih >= 0);
        assert(ih < int(nh));
        if constexpr (CountOrFill::count == cof)
          (*h).count(v[i], ih);
        else
          (*h).fill(v[i], i, ih);
      }
    }

    template <typename Histo, typename T, CountOrFill cof>
    __global__ void countOrFillFromVectorKernel(Histo *__restrict__ h,
                                                uint32_t nh,
                                                T const *__restrict__ v,
                                                uint32_t const *__restrict__ offsets) {
      countOrFillFromVector<Histo, T, cof>(h, nh, v, offsets);
    }

    template <typename Histo, typename T>
    inline __attribute__((always_inline)) void fillManyFromVector(Histo *__restrict__ h,
                                                                  uint32_t nh,
                                                                  T const *__restrict__ v,
                                                                  uint32_t const *__restrict__ offsets,
                                                                  int32_t totSize,
                                                                  int nthreads,
                                                                  typename Histo::index_type *mem,
                                                                  cudaStream_t stream
#ifndef __CUDACC__
                                                                  = cudaStreamDefault
#endif
    ) {
      typename Histo::View view = {h, nullptr, mem, -1, totSize};
      launchZero(view, stream);
#ifdef __CUDACC__
      auto nblocks = (totSize + nthreads - 1) / nthreads;
      assert(nblocks > 0);
      countOrFillFromVectorKernel<Histo, T, CountOrFill::count><<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
      launchFinalize(view, stream);
      countOrFillFromVectorKernel<Histo, T, CountOrFill::fill><<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
      cudaCheck(cudaGetLastError());
#else
      countOrFillFromVectorKernel<Histo, T, CountOrFill::count>(h, nh, v, offsets);
      h->finalize();
      countOrFillFromVectorKernel<Histo, T, CountOrFill::fill>(h, nh, v, offsets);
#endif
    }

#ifdef __CUDACC__
    template <typename Histo, typename T>
    __global__ void fillManyFromVectorCoopKernel(typename Histo::View view,
                                                 uint32_t nh,
                                                 T const *__restrict__ v,
                                                 uint32_t const *__restrict__ offsets,
                                                 int32_t totSize,
                                                 typename Histo::View::Counter *ws) {
      namespace cg = cooperative_groups;
      auto grid = cg::this_grid();
      auto h = static_cast<Histo *>(view.assoc);
      zeroAndInitCoop(view);
      grid.sync();
      countOrFillFromVector<Histo, T, CountOrFill::count>(h, nh, v, offsets);
      grid.sync();
      finalizeCoop(view, ws);
      grid.sync();
      countOrFillFromVector<Histo, T, CountOrFill::fill>(h, nh, v, offsets);
    }
#endif

    template <typename Histo, typename T>
    inline __attribute__((always_inline)) void fillManyFromVectorCoop(Histo *h,
                                                                      uint32_t nh,
                                                                      T const *v,
                                                                      uint32_t const *offsets,
                                                                      int32_t totSize,
                                                                      int nthreads,
                                                                      typename Histo::index_type *mem,
                                                                      cudaStream_t stream
#ifndef __CUDACC__
                                                                      = cudaStreamDefault
#endif
    ) {
      using View = typename Histo::View;
      View view = {h, nullptr, mem, -1, totSize};
#ifdef __CUDACC__
      auto kernel = fillManyFromVectorCoopKernel<Histo, T>;
      auto nblocks = (totSize + nthreads - 1) / nthreads;
      assert(nblocks > 0);
      auto nOnes = view.size();
      auto nchunks = nOnes / nthreads + 1;
      auto ws = cms::cuda::make_device_unique<typename View::Counter[]>(nchunks, stream);
      auto wsp = ws.get();
      // FIXME: discuss with FW team: cuda calls are expensive and not needed for each event
      static int maxBlocks = maxCoopBlocks(kernel, nthreads, 0, 0);
      auto ncoopblocks = std::min(nblocks, maxBlocks);
      assert(ncoopblocks > 0);
      void *kernelArgs[] = {&view, &nh, &v, &offsets, &totSize, &wsp};
      dim3 dimBlock(nthreads, 1, 1);
      dim3 dimGrid(ncoopblocks, 1, 1);
      // launch
      cudaCheck(cudaLaunchCooperativeKernel((void *)kernel, dimGrid, dimBlock, kernelArgs, 0, stream));
      cudaCheck(cudaGetLastError());
#else
      launchZero(view, stream);
      countFromVector(h, nh, v, offsets);
      h->finalize();
      fillFromVector(h, nh, v, offsets);
#endif
    }

    // iteratate over N bins left and right of the one containing "v"
    template <typename Hist, typename V, typename Func>
    __host__ __device__ __forceinline__ void forEachInBins(Hist const &hist, V value, int n, Func func) {
      int bs = Hist::bin(value);
      int be = std::min(int(Hist::nbins() - 1), bs + n);
      bs = std::max(0, bs - n);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }

    // iteratate over bins containing all values in window wmin, wmax
    template <typename Hist, typename V, typename Func>
    __host__ __device__ __forceinline__ void forEachInWindow(Hist const &hist, V wmin, V wmax, Func &&func) {
      auto bs = Hist::bin(wmin);
      auto be = Hist::bin(wmax);
      assert(be >= bs);
      for (auto pj = hist.begin(bs); pj < hist.end(be); ++pj) {
        func(*pj);
      }
    }
  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_HistoContainerAlgo_h
