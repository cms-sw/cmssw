#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "EcalUncalibRecHitPhase2WeightsKernels.h"
#include "EcalUncalibRecHitPhase2WeightsAlgoGPU.h"

namespace ecal {
  namespace weights {

    void entryPoint(ecal::DigisCollection<calo::common::DevStoragePolicy> const& digis,
                    EventOutputDataGPU& eventOutputGPU,
                    cms::cuda::device::unique_ptr<double[]>& weights_d,
                    cudaStream_t cudaStream) {
      unsigned int totalChannels = digis.size;
      // 64 threads per block best occupancy from Nsight compute profiler
      unsigned int nchannels_per_block = 64;
      unsigned int threads_1d = nchannels_per_block;
      unsigned int blocks_1d = (totalChannels / threads_1d) + 1;
      // shared bytes from size of weight constants, digi samples per block, uncalib rechits amplitudes per block
      int shared_bytes = EcalDataFrame_Ph2::MAXSAMPLES * sizeof(double) +
                         nchannels_per_block * (EcalDataFrame_Ph2::MAXSAMPLES * (sizeof(uint16_t)) + sizeof(float));
      Phase2WeightsKernel<<<blocks_1d, threads_1d, shared_bytes, cudaStream>>>(
          digis.data.get(),
          digis.ids.get(),
          eventOutputGPU.recHits.amplitude.get(),
          eventOutputGPU.recHits.amplitudeError.get(),
          eventOutputGPU.recHits.did.get(),
          totalChannels,
          weights_d.get(),
          eventOutputGPU.recHits.flags.get());
      cudaCheck(cudaGetLastError());
    }

  }  // namespace weights
}  // namespace ecal