#include <iostream>
#include <limits>

#include <cuda.h>
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "EcalUncalibRecHitPhase2WeightsKernels.h"
#include "EcalUncalibRecHitPhase2WeightsAlgoGPU.h"

#include "EigenMatrixTypes_gpu.h"

#include "DeclsForKernelsPh2WeightsGPU.h"

// entrypoint to kernal execution

//#define DEBUG

//#define ECAL_RECO_CUDA_DEBUG

namespace ecal {
  namespace weights {

    void entryPoint(ecal::DigisCollection<calo::common::DevStoragePolicy> const& ebDigis,
                    EventOutputDataGPUWeights& eventOutputGPU,
                    cms::cuda::device::unique_ptr<double[]>& weights_d,
                    cudaStream_t cudaStream) {
      unsigned int totalChannels = ebDigis.size;

      unsigned int nchannels_per_block = 64;
      unsigned int threads_1d = nchannels_per_block;
      unsigned int blocks_1d = (totalChannels / threads_1d) + 1;

      int shared_bytes = 2 * sizeof(float) + EcalDataFrame_Ph2::MAXSAMPLES * sizeof(double) +
                         nchannels_per_block * (EcalDataFrame_Ph2::MAXSAMPLES * (sizeof(uint16_t)) + sizeof(float));

      Phase2WeightsKernel<<<blocks_1d, threads_1d, shared_bytes, cudaStream>>>(
          ebDigis.data.get(),
          ebDigis.ids.get(),
          eventOutputGPU.recHitsEB.amplitude.get(),
          eventOutputGPU.recHitsEB.amplitudeError.get(),
          eventOutputGPU.recHitsEB.did.get(),
          totalChannels,
          weights_d.get(),
          eventOutputGPU.recHitsEB.flags.get());
      cudaCheck(cudaGetLastError());
    }

  }  // namespace weights
}  // namespace ecal
