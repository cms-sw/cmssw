// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"
#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  __global__
  void setHitsLayerStart(const uint32_t* hitsModuleStart, const uint32_t* layerStart, uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < 10) {
      hitsLayerStart[i] = hitsModuleStart[layerStart[i]];
    }
    else if(i == 10) {
      hitsLayerStart[i] = hitsModuleStart[gpuClustering::MaxNumModules];
    }
  }

  template <typename T>
  T *slicePitch(void *ptr, size_t pitch, size_t row) {
    return reinterpret_cast<T *>( reinterpret_cast<char *>(ptr) + pitch*row);
  }
}

namespace pixelgpudetails {
  PixelRecHitGPUKernel::PixelRecHitGPUKernel(cuda::stream_t<>& cudaStream) {

    constexpr auto MAX_HITS = siPixelRecHitsHeterogeneousProduct::maxHits();

    cudaCheck(cudaMalloc((void **) & gpu_.bs_d, 3 * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.hitsLayerStart_d, 11 * sizeof(uint32_t)));

    // Coalesce all 32bit and 16bit arrays to two big blobs
    //
    // This is just a toy. Please don't copy-paste the logic but
    // create a proper abstraction (e.g. along FWCore/SOA, or
    // FWCore/Utilities/interface/SoATuple.h
    //
    // Order such that the first ones are the ones transferred to CPU
    static_assert(sizeof(uint32_t) == sizeof(float)); // just stating the obvious
    cudaCheck(cudaMallocPitch(&gpu_.owner_32bit_, &gpu_.owner_32bit_pitch_, MAX_HITS*sizeof(uint32_t), 9));
    cudaCheck(cudaMemsetAsync(gpu_.owner_32bit_, 0x0, gpu_.owner_32bit_pitch_*9, cudaStream.id()));
    //edm::LogPrint("Foo") << "Allocate 32bit with pitch " << gpu_.owner_32bit_pitch_;
    gpu_.charge_d = slicePitch<int32_t>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 0);
    gpu_.xl_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 1);
    gpu_.yl_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 2);
    gpu_.xerr_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 3);
    gpu_.yerr_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 4);
    gpu_.xg_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 5);
    gpu_.yg_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 6);
    gpu_.zg_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 7);
    gpu_.rg_d = slicePitch<float>(gpu_.owner_32bit_, gpu_.owner_32bit_pitch_, 8);

    // Order such that the first ones are the ones transferred to CPU
    cudaCheck(cudaMallocPitch(&gpu_.owner_16bit_, &gpu_.owner_16bit_pitch_, MAX_HITS*sizeof(uint16_t), 5));
    cudaCheck(cudaMemsetAsync(gpu_.owner_16bit_, 0x0, gpu_.owner_16bit_pitch_*5, cudaStream.id()));
    //edm::LogPrint("Foo") << "Allocate 16bit with pitch " << gpu_.owner_16bit_pitch_;
    gpu_.detInd_d = slicePitch<uint16_t>(gpu_.owner_16bit_, gpu_.owner_16bit_pitch_, 0);
    gpu_.mr_d = slicePitch<uint16_t>(gpu_.owner_16bit_, gpu_.owner_16bit_pitch_, 1);
    gpu_.mc_d = slicePitch<uint16_t>(gpu_.owner_16bit_, gpu_.owner_16bit_pitch_, 2);
    gpu_.iphi_d = slicePitch<int16_t>(gpu_.owner_16bit_, gpu_.owner_16bit_pitch_, 3);
    gpu_.sortIndex_d = slicePitch<uint16_t>(gpu_.owner_16bit_, gpu_.owner_16bit_pitch_, 4);

    cudaCheck(cudaMalloc((void **) & gpu_.hist_d, sizeof(HitsOnGPU::Hist)));
    cudaCheck(cudaMalloc((void **) & gpu_.hws_d, HitsOnGPU::Hist::wsSize()));
    cudaCheck(cudaMalloc((void **) & gpu_d, sizeof(HitsOnGPU)));

    // Feels a bit dumb but constexpr arrays are not supported for device code
    // TODO: should be moved to EventSetup (or better ideas?)
    // Would it be better to use "constant memory"?
    cudaCheck(cudaMalloc((void **) & d_phase1TopologyLayerStart_, 11 * sizeof(uint32_t)));
    cudaCheck(cudaMemcpyAsync(d_phase1TopologyLayerStart_, phase1PixelTopology::layerStart, 11 * sizeof(uint32_t), cudaMemcpyDefault, cudaStream.id()));
    cudaCheck(cudaMalloc((void **) & d_phase1TopologyLayer_, phase1PixelTopology::layer.size() * sizeof(uint8_t)));
    cudaCheck(cudaMemcpyAsync(d_phase1TopologyLayer_, phase1PixelTopology::layer.data(), phase1PixelTopology::layer.size() * sizeof(uint8_t), cudaMemcpyDefault, cudaStream.id()));

    gpu_.phase1TopologyLayerStart_d = d_phase1TopologyLayerStart_;
    gpu_.phase1TopologyLayer_d = d_phase1TopologyLayer_;

    gpu_.me_d = gpu_d;
    cudaCheck(cudaMemcpyAsync(gpu_d, &gpu_, sizeof(HitsOnGPU), cudaMemcpyDefault, cudaStream.id()));

    cudaCheck(cudaMallocHost(&h_hitsModuleStart_, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t)));

    // On CPU we can safely use MAX_HITS*sizeof as the pitch. Thanks
    // to '*256' it is even aligned by cache line
    h_owner_32bit_pitch_ = MAX_HITS*sizeof(uint32_t); 
    cudaCheck(cudaMallocHost(&h_owner_32bit_, h_owner_32bit_pitch_ * 5));
    h_charge_ = slicePitch<int32_t>(h_owner_32bit_, h_owner_32bit_pitch_, 0);
    h_xl_ = slicePitch<float>(h_owner_32bit_, h_owner_32bit_pitch_, 1);
    h_yl_ = slicePitch<float>(h_owner_32bit_, h_owner_32bit_pitch_, 2);
    h_xe_ = slicePitch<float>(h_owner_32bit_, h_owner_32bit_pitch_, 3);
    h_ye_ = slicePitch<float>(h_owner_32bit_, h_owner_32bit_pitch_, 4);

    h_owner_16bit_pitch_ = MAX_HITS*sizeof(uint16_t);
    cudaCheck(cudaMallocHost(&h_owner_16bit_, h_owner_16bit_pitch_ * 3));
    h_detInd_ = slicePitch<uint16_t>(h_owner_16bit_, h_owner_16bit_pitch_, 0);
    h_mr_ = slicePitch<uint16_t>(h_owner_16bit_, h_owner_16bit_pitch_, 1);
    h_mc_ = slicePitch<uint16_t>(h_owner_16bit_, h_owner_16bit_pitch_, 2);

#ifdef GPU_DEBUG
    cudaCheck(cudaMallocHost(&h_hitsLayerStart_, 11 * sizeof(uint32_t)));
#endif
  }
  PixelRecHitGPUKernel::~PixelRecHitGPUKernel() {
    cudaCheck(cudaFree(gpu_.bs_d));
    cudaCheck(cudaFree(gpu_.hitsLayerStart_d));
    cudaCheck(cudaFree(gpu_.owner_32bit_));
    cudaCheck(cudaFree(gpu_.owner_16bit_));
    cudaCheck(cudaFree(gpu_.hist_d));
    cudaCheck(cudaFree(gpu_.hws_d));
    cudaCheck(cudaFree(gpu_d));
    cudaCheck(cudaFree(d_phase1TopologyLayerStart_));
    cudaCheck(cudaFree(d_phase1TopologyLayer_));

    cudaCheck(cudaFreeHost(h_hitsModuleStart_));
    cudaCheck(cudaFreeHost(h_owner_32bit_));
    cudaCheck(cudaFreeHost(h_owner_16bit_));
#ifdef GPU_DEBUG
    cudaCheck(cudaFreeHost(h_hitsLayerStart_));
#endif
  }

  void PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                           SiPixelClustersCUDA const& clusters_d,
                                           float const * bs,
                                           pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                                           bool transferToCPU,
                                           cuda::stream_t<>& stream) {
    cudaCheck(cudaMemcpyAsync(gpu_.bs_d, bs, 3 * sizeof(float), cudaMemcpyDefault, stream.id()));
    gpu_.hitsModuleStart_d = clusters_d.clusModuleStart();
    gpu_.cpeParams = cpeParams; // copy it for use in clients
    cudaCheck(cudaMemcpyAsync(gpu_d, &gpu_, sizeof(HitsOnGPU), cudaMemcpyDefault, stream.id()));

    int threadsPerBlock = 256;
    int blocks = digis_d.nModules(); // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream.id()>>>(
      cpeParams,
      gpu_.bs_d,
      digis_d.moduleInd(),
      digis_d.xx(), digis_d.yy(), digis_d.adc(),
      clusters_d.moduleStart(),
      clusters_d.clusInModule(), clusters_d.moduleId(),
      digis_d.clus(),
      digis_d.nDigis(),
      gpu_.hitsModuleStart_d,
      gpu_.charge_d,
      gpu_.detInd_d,
      gpu_.xg_d, gpu_.yg_d, gpu_.zg_d, gpu_.rg_d,
      gpu_.iphi_d,
      gpu_.xl_d, gpu_.yl_d,
      gpu_.xerr_d, gpu_.yerr_d,
      gpu_.mr_d, gpu_.mc_d
    );
    cudaCheck(cudaGetLastError());

    // assuming full warp of threads is better than a smaller number...
    setHitsLayerStart<<<1, 32, 0, stream.id()>>>(gpu_.hitsModuleStart_d, d_phase1TopologyLayerStart_, gpu_.hitsLayerStart_d);
    cudaCheck(cudaGetLastError());

    // needed only if hits on CPU are required...
    nhits_ = clusters_d.nClusters();
    if(transferToCPU) {
      cudaCheck(cudaMemcpyAsync(h_hitsModuleStart_, gpu_.hitsModuleStart_d, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
#ifdef GPU_DEBUG
      cudaCheck(cudaMemcpyAsync(h_hitsLayerStart_, gpu_.hitsLayerStart_d, 11 * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
#endif

      cudaCheck(cudaMemcpy2DAsync(h_owner_16bit_, h_owner_16bit_pitch_,
                                  gpu_.owner_16bit_, gpu_.owner_16bit_pitch_,
                                  nhits_*sizeof(uint16_t), 3,
                                  cudaMemcpyDefault, stream.id()));

      cudaCheck(cudaMemcpy2DAsync(h_owner_32bit_, h_owner_32bit_pitch_,
                                  gpu_.owner_32bit_, gpu_.owner_32bit_pitch_,
                                  nhits_*sizeof(uint32_t), 5,
                                  cudaMemcpyDefault, stream.id()));

#ifdef GPU_DEBUG
      cudaStreamSynchronize(stream.id());

      std::cout << "hit layerStart ";
      for (int i=0;i<10;++i) std::cout << phase1PixelTopology::layerName[i] << ':' << h_hitsLayerStart_[i] << ' ';
      std::cout << "end:" << h_hitsLayerStart_[10] << std::endl;
#endif

      // for timing test
      // cudaStreamSynchronize(stream.id());
      // auto nhits_ = h_hitsLayerStart_[10];
      // radixSortMultiWrapper<int16_t><<<10, 256, 0, c.stream>>>(gpu_.iphi_d, gpu_.sortIndex_d, gpu_.hitsLayerStart_d);
    }

    cudautils::fillManyFromVector(gpu_.hist_d, gpu_.hws_d, 10, gpu_.iphi_d, gpu_.hitsLayerStart_d, nhits_, 256, stream.id());
  }
}
