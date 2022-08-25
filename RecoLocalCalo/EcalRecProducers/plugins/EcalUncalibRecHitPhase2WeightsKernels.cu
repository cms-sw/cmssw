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
      int const tx = threadIdx.x + blockIdx.x * blockDim.x;
      unsigned int nchannels_per_block = blockDim.x;
      unsigned int const threadx = threadIdx.x;

      if (tx < nchannels) {
        extern __shared__ char shared_mem[];
        double* shr_weights = (double*)&shared_mem[0];
        float* shr_amp = (float*)&shared_mem[nsamples * sizeof(double)];
        uint16_t* shr_digis = (uint16_t*)&shared_mem[nsamples * sizeof(double) + nchannels_per_block * sizeof(float)];
        for (int i = 0; i < nsamples; ++i)
          shr_weights[i] = weights[i];

        unsigned int const bx = blockIdx.x;  //block index

        for (int sample = 0; sample < nsamples; ++sample) {
          int const idx = threadx * nsamples + sample;
          shr_digis[idx] = digis_in[bx * nchannels_per_block * nsamples + idx];
        }
        shr_amp[threadx] = 0.0;
        __syncthreads();

        auto const did = DetId{dids[tx]};
        CMS_UNROLL_LOOP
        for (int sample = 0; sample < nsamples; ++sample) {
          const unsigned int idx = threadIdx.x * nsamples + sample;
          const auto shr_digi = shr_digis[idx];
          shr_amp[threadx] += (static_cast<float>(ecalLiteDTU::adc(shr_digi)) *
                               ecalPh2::gains[ecalLiteDTU::gainId(shr_digi)] * shr_weights[sample]);
        }
        const unsigned int tdx = threadIdx.x * nsamples;
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
