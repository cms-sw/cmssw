#include <cuda.h>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include "EcalUncalibRecHitPhase2WeightsKernels.h"

namespace ecal {
  namespace weights {

    __global__ void Phase2WeightsKernel(uint16_t const* digis_in,
                                        uint32_t const* __restrict__ dids,
                                        ::ecal::reco::StorageScalarType* __restrict__ amplitude,
                                        ::ecal::reco::StorageScalarType* __restrict__ amplitudeError,
                                        uint32_t* __restrict__ dids_out,
                                        int const nchannels,
                                        double const* __restrict__ weights,
                                        uint32_t* __restrict__ flags) {
      constexpr int nsamples = EcalDataFrame_Ph2::MAXSAMPLES;
      unsigned int nchannels_per_block = blockDim.x;

      // copy data from global to shared memory
      extern __shared__ char shared_mem[];
      double* shr_weights = reinterpret_cast<double*>(shared_mem);                       // nsamples elements
      float* shr_amp = reinterpret_cast<float*>(shr_weights + nsamples);                 // nchannels_per_block elements
      uint16_t* shr_digis = reinterpret_cast<uint16_t*>(shr_amp + nchannels_per_block);  // nchannels_per_block elements
      for (int i = 0; i < nsamples; ++i)
        shr_weights[i] = weights[i];

      unsigned int const threadx = threadIdx.x;
      unsigned int const blockx = blockIdx.x;

      for (int sample = 0; sample < nsamples; ++sample) {
        int const idx = threadx * nsamples + sample;
        shr_digis[idx] = digis_in[blockx * nchannels_per_block * nsamples + idx];
      }
      shr_amp[threadx] = 0.;

      __syncthreads();

      const auto first = threadIdx.x + blockIdx.x * blockDim.x;
      const auto stride = blockDim.x * gridDim.x;
      for (auto tx = first; tx < nchannels; tx += stride) {
        auto const did = DetId{dids[tx]};
        CMS_UNROLL_LOOP
        for (int sample = 0; sample < nsamples; ++sample) {
          const unsigned int idx = threadx * nsamples + sample;
          const auto shr_digi = shr_digis[idx];
          shr_amp[threadx] += (static_cast<float>(ecalLiteDTU::adc(shr_digi)) *
                               ecalPh2::gains[ecalLiteDTU::gainId(shr_digi)] * shr_weights[sample]);
        }
        const unsigned int tdx = threadx * nsamples;
        amplitude[tx] = shr_amp[threadx];
        amplitudeError[tx] = 1.0f;
        dids_out[tx] = did.rawId();
        flags[tx] = 0;
        if (ecalLiteDTU::gainId(shr_digis[tdx + nsamples - 1])) {
          flags[tx] = 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;
        }
      }  //if within nchannels
    }    //kernel
  }      //namespace weights
}  //namespace ecal
