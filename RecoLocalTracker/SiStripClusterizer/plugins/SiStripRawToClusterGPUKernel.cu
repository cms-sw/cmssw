#include <cub/cub.cuh>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"

#include "ChannelLocsGPU.h"
#include "SiStripRawToClusterGPUKernel.h"
#include "StripDataView.h"

//#define GPU_DEBUG
#if defined(EDM_ML_DEBUG) || defined(GPU_DEBUG)
#define GPU_CHECK
#include <cstdio>
#endif

using namespace stripgpu;
using ConditionsDeviceView = SiStripClusterizerConditionsGPU::Data::DeviceView;

__global__ static void unpackChannels(const ChannelLocsView *chanlocs,
                                      const ConditionsDeviceView *conditions,
                                      uint8_t *alldata,
                                      uint16_t *channel,
                                      stripId_t *stripId) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  const auto first = nthreads * bid + tid;
  const auto stride = blockDim.x * gridDim.x;
  for (auto chan = first; chan < chanlocs->size(); chan += stride) {
    const auto fedid = chanlocs->fedID(chan);
    const auto fedch = chanlocs->fedCh(chan);
    const auto ipair = conditions->iPair(fedid, fedch);
    const auto ipoff = sistrip::STRIPS_PER_FEDCH * ipair;

    const auto data = chanlocs->input(chan);
    const auto len = chanlocs->length(chan);

    if (data != nullptr && len > 0) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);
      const auto end = choff + len;

      while (choff < end) {
        auto stripIndex = data[(choff++) ^ 7] + ipoff;
        const auto groupLength = data[(choff++) ^ 7];

        for (auto i = 0; i < 2; ++i) {
          stripId[aoff] = invalidStrip;
          alldata[aoff++] = 0;
        }

        for (auto i = 0; i < groupLength; ++i) {
          stripId[aoff] = stripIndex++;
          channel[aoff] = chan;
          alldata[aoff++] = data[(choff++) ^ 7];
        }
      }
    }  // choff < end
  }    // data != nullptr && len > 0
}  // chan < chanlocs->size()

__global__ static void setSeedStripsGPU(StripDataView *sst_data_d, const ConditionsDeviceView *conditions) {
  const int nStrips = sst_data_d->nStrips;
  const auto __restrict__ chanlocs = sst_data_d->chanlocs;
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const uint16_t *__restrict__ channels = sst_data_d->channel;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;
  const float seedThreshold = sst_data_d->seedThreshold;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int first = nthreads * bid + tid;
  const int stride = blockDim.x * gridDim.x;

  for (int i = first; i < nStrips; i += stride) {
    seedStripsMask[i] = 0;
    seedStripsNCMask[i] = 0;
    const stripId_t strip = stripId[i];
    if (strip != invalidStrip) {
      const auto chan = channels[i];
      const fedId_t fed = chanlocs->fedID(chan);
      const fedCh_t channel = chanlocs->fedCh(chan);
      const float noise_i = conditions->noise(fed, channel, strip);
      const uint8_t adc_i = adc[i];

      seedStripsMask[i] = (adc_i >= static_cast<uint8_t>(noise_i * seedThreshold)) ? 1 : 0;
      seedStripsNCMask[i] = seedStripsMask[i];
    }
  }
}

__global__ static void setNCSeedStripsGPU(StripDataView *sst_data_d, const ConditionsDeviceView *conditions) {
  const int nStrips = sst_data_d->nStrips;
  const auto __restrict__ chanlocs = sst_data_d->chanlocs;
  const uint16_t *__restrict__ channels = sst_data_d->channel;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int first = nthreads * bid + tid;
  const int stride = blockDim.x * gridDim.x;

  for (int i = first; i < nStrips; i += stride) {
    if (i > 0) {
      const auto detid = chanlocs->detID(channels[i]);
      const auto detid1 = chanlocs->detID(channels[i - 1]);

      if (seedStripsMask[i] && seedStripsMask[i - 1] && (stripId[i] - stripId[i - 1]) == 1 && (detid == detid1))
        seedStripsNCMask[i] = 0;
    }
  }
}

__global__ static void setStripIndexGPU(StripDataView *sst_data_d) {
  const int nStrips = sst_data_d->nStrips;
  const int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;
  const int *__restrict__ prefixSeedStripsNCMask = sst_data_d->prefixSeedStripsNCMask;
  int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int first = nthreads * bid + tid;
  const int stride = blockDim.x * gridDim.x;

  for (int i = first; i < nStrips; i += stride) {
    if (seedStripsNCMask[i] == 1) {
      const int index = prefixSeedStripsNCMask[i];
      seedStripsNCIndex[index] = i;
    }
  }
}

__global__ static void findLeftRightBoundaryGPU(const StripDataView *sst_data_d,
                                                const ConditionsDeviceView *conditions,
                                                SiStripClustersCUDADevice::DeviceView *clust_data_d) {
  const int nStrips = sst_data_d->nStrips;
  const int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;
  const auto __restrict__ chanlocs = sst_data_d->chanlocs;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const uint16_t *__restrict__ channels = sst_data_d->channel;
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const int nSeedStripsNC = std::min(kMaxSeedStrips, *(sst_data_d->prefixSeedStripsNCMask + nStrips - 1));
  const uint8_t maxSequentialHoles = sst_data_d->maxSequentialHoles;
  const float channelThreshold = sst_data_d->channelThreshold;
  const float clusterThresholdSquared = sst_data_d->clusterThresholdSquared;
  const int clusterSizeLimit = sst_data_d->clusterSizeLimit;

  auto __restrict__ clusterIndexLeft = clust_data_d->clusterIndex_;
  auto __restrict__ clusterSize = clust_data_d->clusterSize_;
  auto __restrict__ clusterDetId = clust_data_d->clusterDetId_;
  auto __restrict__ firstStrip = clust_data_d->firstStrip_;
  auto __restrict__ trueCluster = clust_data_d->trueCluster_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int first = nthreads * bid + tid;
  const int stride = blockDim.x * gridDim.x;

  for (int i = first; i < nSeedStripsNC; i += stride) {
    const auto index = seedStripsNCIndex[i];
    const auto chan = channels[index];
    const auto fed = chanlocs->fedID(chan);
    const auto channel = chanlocs->fedCh(chan);
    const auto det = chanlocs->detID(chan);
    const auto strip = stripId[index];
    const auto noise_i = conditions->noise(fed, channel, strip);

    auto noiseSquared_i = noise_i * noise_i;
    float adcSum_i = static_cast<float>(adc[index]);
    auto testIndex = index - 1;
    auto size = 1;

    auto addtocluster = [&](int &indexLR) {
      const auto testchan = channels[testIndex];
      const auto testFed = chanlocs->fedID(testchan);
      const auto testChannel = chanlocs->fedCh(testchan);
      const auto testStrip = stripId[testIndex];
      const auto testNoise = conditions->noise(testFed, testChannel, testStrip);
      const auto testADC = adc[testIndex];

      if (testADC >= static_cast<uint8_t>(testNoise * channelThreshold)) {
        ++size;
        indexLR = testIndex;
        noiseSquared_i += testNoise * testNoise;
        adcSum_i += static_cast<float>(testADC);
      }
    };

    // find left boundary
    auto indexLeft = index;

    if (testIndex >= 0 && stripId[testIndex] == invalidStrip) {
      testIndex -= 2;
    }

    if (testIndex >= 0) {
      const auto testchan = channels[testIndex];
      const auto testDet = chanlocs->detID(testchan);
      auto rangeLeft = stripId[indexLeft] - stripId[testIndex] - 1;
      auto sameDetLeft = det == testDet;

      while (sameDetLeft && rangeLeft >= 0 && rangeLeft <= maxSequentialHoles && size < clusterSizeLimit + 1) {
        addtocluster(indexLeft);
        --testIndex;
        if (testIndex >= 0 && stripId[testIndex] == invalidStrip) {
          testIndex -= 2;
        }
        if (testIndex >= 0) {
          rangeLeft = stripId[indexLeft] - stripId[testIndex] - 1;
          const auto newchan = channels[testIndex];
          const auto newdet = chanlocs->detID(newchan);
          sameDetLeft = det == newdet;
        } else {
          sameDetLeft = false;
        }
      }  // while loop
    }    // testIndex >= 0

    // find right boundary
    auto indexRight = index;
    testIndex = index + 1;

    if (testIndex < nStrips && stripId[testIndex] == invalidStrip) {
      testIndex += 2;
    }

    if (testIndex < nStrips) {
      const auto testchan = channels[testIndex];
      const auto testDet = chanlocs->detID(testchan);
      auto rangeRight = stripId[testIndex] - stripId[indexRight] - 1;
      auto sameDetRight = det == testDet;

      while (sameDetRight && rangeRight >= 0 && rangeRight <= maxSequentialHoles && size < clusterSizeLimit + 1) {
        addtocluster(indexRight);
        ++testIndex;
        if (testIndex < nStrips && stripId[testIndex] == invalidStrip) {
          testIndex += 2;
        }
        if (testIndex < nStrips) {
          rangeRight = stripId[testIndex] - stripId[indexRight] - 1;
          const auto newchan = channels[testIndex];
          const auto newdet = chanlocs->detID(newchan);
          sameDetRight = det == newdet;
        } else {
          sameDetRight = false;
        }
      }  // while loop
    }    // testIndex < nStrips
    clusterIndexLeft[i] = indexLeft;
    clusterSize[i] = indexRight - indexLeft + 1;
    clusterDetId[i] = det;
    firstStrip[i] = stripId[indexLeft];
    trueCluster[i] =
        (noiseSquared_i * clusterThresholdSquared <= adcSum_i * adcSum_i) and (clusterSize[i] <= clusterSizeLimit);
  }  // i < nSeedStripsNC
  if (first == 0) {
    clust_data_d->nClusters_ = nSeedStripsNC;
  }
}

__global__ static void checkClusterConditionGPU(StripDataView *sst_data_d,
                                                const ConditionsDeviceView *conditions,
                                                SiStripClustersCUDADevice::DeviceView *clust_data_d) {
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const auto __restrict__ chanlocs = sst_data_d->chanlocs;
  const uint16_t *__restrict__ channels = sst_data_d->channel;
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const float minGoodCharge = sst_data_d->minGoodCharge;  //1620.0;
  const auto nSeedStripsNC = clust_data_d->nClusters_;
  const auto __restrict__ clusterIndexLeft = clust_data_d->clusterIndex_;

  auto __restrict__ clusterSize = clust_data_d->clusterSize_;
  auto __restrict__ clusterADCs = clust_data_d->clusterADCs_;
  auto __restrict__ trueCluster = clust_data_d->trueCluster_;
  auto __restrict__ barycenter = clust_data_d->barycenter_;
  auto __restrict__ charge = clust_data_d->charge_;

  constexpr uint16_t stripIndexMask = 0x7FFF;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int first = nthreads * bid + tid;
  const int stride = blockDim.x * gridDim.x;

  for (int i = first; i < nSeedStripsNC; i += stride) {
    if (trueCluster[i]) {
      const int left = clusterIndexLeft[i];
      const int size = clusterSize[i];

      if (i > 0 && clusterIndexLeft[i - 1] == left) {
        trueCluster[i] = 0;  // ignore duplicates
      } else {
        float adcSum = 0.0f;
        int sumx = 0;
        int suma = 0;

        auto j = 0;
        for (int k = 0; k < size; k++) {
          const auto index = left + k;
          const auto chan = channels[index];
          const auto fed = chanlocs->fedID(chan);
          const auto channel = chanlocs->fedCh(chan);
          const auto strip = stripId[index];
#ifdef GPU_CHECK
          if (fed == invalidFed) {
            printf("Invalid fed index %d\n", index);
          }
#endif
          if (strip != invalidStrip) {
            const float gain_j = conditions->gain(fed, channel, strip);

            uint8_t adc_j = adc[index];
            const int charge = static_cast<int>(static_cast<float>(adc_j) / gain_j + 0.5f);

            constexpr uint8_t adc_low_saturation = 254;
            constexpr uint8_t adc_high_saturation = 255;
            constexpr int charge_low_saturation = 253;
            constexpr int charge_high_saturation = 1022;
            if (adc_j < adc_low_saturation) {
              adc_j =
                  (charge > charge_high_saturation ? adc_high_saturation
                                                   : (charge > charge_low_saturation ? adc_low_saturation : charge));
            }
            clusterADCs[j * nSeedStripsNC + i] = adc_j;

            adcSum += static_cast<float>(adc_j);
            sumx += j * adc_j;
            suma += adc_j;
            j++;
          }
        }  // loop over cluster strips
        charge[i] = adcSum;
        const auto chan = channels[left];
        const fedId_t fed = chanlocs->fedID(chan);
        const fedCh_t channel = chanlocs->fedCh(chan);
        trueCluster[i] = (adcSum * conditions->invthick(fed, channel)) > minGoodCharge;
        const auto bary_i = static_cast<float>(sumx) / static_cast<float>(suma);
        barycenter[i] = static_cast<float>(stripId[left] & stripIndexMask) + bary_i + 0.5f;
        clusterSize[i] = j;
      }  // not a duplicate cluster
    }    // trueCluster[i] is true
  }      // i < nSeedStripsNC
}

namespace stripgpu {
  void SiStripRawToClusterGPUKernel::unpackChannelsGPU(const ConditionsDeviceView *conditions, cudaStream_t stream) {
    constexpr int nthreads = 128;
    const auto channels = chanlocsGPU_->size();
    const auto nblocks = (channels + nthreads - 1) / nthreads;

    unpackChannels<<<nblocks, nthreads, 0, stream>>>(chanlocsGPU_->channelLocsView(),
                                                     conditions,
                                                     stripdata_->alldataGPU_.get(),
                                                     stripdata_->channelGPU_.get(),
                                                     stripdata_->stripIdGPU_.get());
  }

  void SiStripRawToClusterGPUKernel::allocateSSTDataGPU(int max_strips, cudaStream_t stream) {
    stripdata_->seedStripsMask_ = cms::cuda::make_device_unique<int[]>(2 * max_strips, stream);
    stripdata_->prefixSeedStripsNCMask_ = cms::cuda::make_device_unique<int[]>(2 * max_strips, stream);

    sst_data_d_->chanlocs = chanlocsGPU_->channelLocsView();
    sst_data_d_->stripId = stripdata_->stripIdGPU_.get();
    sst_data_d_->channel = stripdata_->channelGPU_.get();
    sst_data_d_->adc = stripdata_->alldataGPU_.get();
    sst_data_d_->seedStripsMask = stripdata_->seedStripsMask_.get();
    sst_data_d_->prefixSeedStripsNCMask = stripdata_->prefixSeedStripsNCMask_.get();

    sst_data_d_->seedStripsNCMask = sst_data_d_->seedStripsMask + max_strips;
    sst_data_d_->seedStripsNCIndex = sst_data_d_->prefixSeedStripsNCMask + max_strips;

    sst_data_d_->channelThreshold = channelThreshold_;
    sst_data_d_->seedThreshold = seedThreshold_;
    sst_data_d_->clusterThresholdSquared = clusterThresholdSquared_;
    sst_data_d_->maxSequentialHoles = maxSequentialHoles_;
    sst_data_d_->maxSequentialBad = maxSequentialBad_;
    sst_data_d_->maxAdjacentBad = maxAdjacentBad_;
    sst_data_d_->minGoodCharge = minGoodCharge_;
    sst_data_d_->clusterSizeLimit = maxClusterSize_;

    pt_sst_data_d_ = cms::cuda::make_device_unique<StripDataView>(stream);
    cms::cuda::copyAsync(pt_sst_data_d_, sst_data_d_, stream);
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif
  }

  void SiStripRawToClusterGPUKernel::findClusterGPU(const ConditionsDeviceView *conditions, cudaStream_t stream) {
    const int nthreads = 128;
    const int nStrips = sst_data_d_->nStrips;
    const int nSeeds = std::min(kMaxSeedStrips, nStrips);
    const int nblocks = (nSeeds + nthreads - 1) / nthreads;

#ifdef GPU_DEBUG
    auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_strip = cms::cuda::make_host_unique<uint16_t[]>(nStrips, stream);
    auto cpu_adc = cms::cuda::make_host_unique<uint8_t[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(
        cpu_strip.get(), sst_data_d_->stripId, nStrips * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(
        cudaMemcpyAsync(cpu_adc.get(), sst_data_d_->adc, nStrips * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_index.get(), sst_data_d_->seedStripsNCIndex, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    for (int i = 0; i < nStrips; i++) {
      std::cout << " cpu_strip " << cpu_strip[i] << " cpu_adc " << (unsigned int)cpu_adc[i] << " cpu index "
                << cpu_index[i] << std::endl;
    }
#endif

    auto clust_data_d = clusters_d_.view();
    findLeftRightBoundaryGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions, clust_data_d);
    cudaCheck(cudaGetLastError());
#ifdef GPU_CHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    cudaCheck(cudaMemcpyAsync(clusters_d_.nClustersPtr(),
                              &(clust_data_d->nClusters_),
                              sizeof(clust_data_d->nClusters_),
                              cudaMemcpyDeviceToHost,
                              stream));

    checkClusterConditionGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions, clust_data_d);
    cudaCheck(cudaGetLastError());

#ifdef GPU_CHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

#ifdef GPU_DEBUG
    cudaStreamSynchronize(stream);
    auto clust_data = std::make_unique<SiStripClustersCUDAHost>(clusters_d_, stream);
    cudaStreamSynchronize(stream);

    const auto clusterIndexLeft = clust_data->clusterIndex().get();
    const auto clusterSize = clust_data->clusterSize().get();
    const auto trueCluster = clust_data->trueCluster().get();
    const auto clusterADCs = clust_data->clusterADCs().get();
    const auto detids = clust_data->clusterDetId().get();
    const auto charge = clust_data->charge().get();

    const auto nSeedStripsNC = clusters_d_.nClusters();
    std::cout << "findClusterGPU nSeedStripsNC=" << nSeedStripsNC << std::endl;

    for (auto i = 0U; i < nSeedStripsNC; i++) {
      if (trueCluster[i]) {
        int left = clusterIndexLeft[i];
        uint32_t size = clusterSize[i];
        const auto detid = detids[i];
        std::cout << "i=" << i << " detId " << detid << " left " << left << " size " << size << " charge " << charge[i]
                  << ": ";
        size = std::min(size, maxClusterSize_);
        for (uint32_t j = 0; j < size; j++) {
          std::cout << (unsigned int)clusterADCs[j * nSeedStripsNC + i] << " ";
        }
        std::cout << std::endl;
      }
    }
#endif
  }

  void SiStripRawToClusterGPUKernel::setSeedStripsNCIndexGPU(const ConditionsDeviceView *conditions,
                                                             cudaStream_t stream) {
#ifdef GPU_DEBUG
    int nStrips = sst_data_d_->nStrips;
    auto cpu_strip = cms::cuda::make_host_unique<uint16_t[]>(nStrips, stream);
    auto cpu_adc = cms::cuda::make_host_unique<uint8_t[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(
        cpu_strip.get(), sst_data_d_->stripId, nStrips * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(
        cudaMemcpyAsync(cpu_adc.get(), sst_data_d_->adc, nStrips * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    for (int i = 0; i < nStrips; i++) {
      std::cout << " cpu_strip " << cpu_strip[i] << " cpu_adc " << (unsigned int)cpu_adc[i] << std::endl;
    }
#endif

    int nthreads = 256;
    int nblocks = (sst_data_d_->nStrips + nthreads - 1) / nthreads;

    //mark seed strips
    setSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions);
    cudaCheck(cudaGetLastError());
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif

    //mark only non-consecutive seed strips (mask out consecutive seed strips)
    setNCSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions);
    cudaCheck(cudaGetLastError());
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif

    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  temp_storage_bytes,
                                  sst_data_d_->seedStripsNCMask,
                                  sst_data_d_->prefixSeedStripsNCMask,
                                  sst_data_d_->nStrips,
                                  stream);
#ifdef GPU_DEBUG
    std::cout << "temp_storage_bytes=" << temp_storage_bytes << std::endl;
#endif
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif

    {
      auto d_temp_storage = cms::cuda::make_device_unique<uint8_t[]>(temp_storage_bytes, stream);
      cub::DeviceScan::ExclusiveSum(d_temp_storage.get(),
                                    temp_storage_bytes,
                                    sst_data_d_->seedStripsNCMask,
                                    sst_data_d_->prefixSeedStripsNCMask,
                                    sst_data_d_->nStrips,
                                    stream);
    }
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif

    setStripIndexGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get());
    cudaCheck(cudaGetLastError());
#ifdef GPU_CHECK
    cudaCheck(cudaStreamSynchronize(stream));
#endif

#ifdef GPU_DEBUG
    auto cpu_mask = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_prefix = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(&(sst_data_d_->nSeedStripsNC),
                              sst_data_d_->prefixSeedStripsNCMask + sst_data_d_->nStrips - 1,
                              sizeof(int),
                              cudaMemcpyDeviceToHost,
                              stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_mask.get(), sst_data_d_->seedStripsNCMask, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_prefix.get(), sst_data_d_->prefixSeedStripsNCMask, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_index.get(), sst_data_d_->seedStripsNCIndex, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    const int nSeedStripsNC = std::min(kMaxSeedStrips, sst_data_d_->nSeedStripsNC);
    std::cout << "nStrips=" << nStrips << " nSeedStripsNC=" << sst_data_d_->nSeedStripsNC << std::endl;
    for (int i = 0; i < nStrips; i++) {
      std::cout << " i " << i << " mask " << cpu_mask[i] << " prefix " << cpu_prefix[i] << " index " << cpu_index[i]
                << std::endl;
    }
#endif
  }
}  // namespace stripgpu
