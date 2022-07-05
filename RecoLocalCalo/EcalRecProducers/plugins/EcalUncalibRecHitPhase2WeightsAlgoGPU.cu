#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "EcalUncalibRecHitPhase2WeightsKernels.h"
#include "EcalUncalibRecHitPhase2WeightsAlgoGPU.h"

namespace ecal {
  namespace weights {

    void phase2Weights(ecal::DigisCollection<calo::common::DevStoragePolicy> const& digis,
                       EventOutputDataGPU& eventOutputGPU,
                       cms::cuda::device::unique_ptr<double[]>& weights_d,
                       cudaStream_t cudaStream) {
      unsigned int const totalChannels = digis.size;
      // 64 threads per block best occupancy from Nsight compute profiler
      unsigned int const threads_1d = 64;
      unsigned int const blocks_1d = (totalChannels + threads_1d - 1) / threads_1d;
      int shared_bytes = EcalDataFrame_Ph2::MAXSAMPLES * sizeof(double) +
                         threads_1d * (EcalDataFrame_Ph2::MAXSAMPLES * (sizeof(uint16_t)) + sizeof(float));
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
