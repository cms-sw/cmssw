#include <cuda.h>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include "EcalUncalibRecHitPhase2WeightsKernels.h"

namespace ecal {
  namespace weights {

    __global__ void Phase2WeightsKernel(uint16_t const* digis_in,
                                        uint32_t const* dids,
                                        ::ecal::reco::StorageScalarType* amplitude,
                                        ::ecal::reco::StorageScalarType* amplitudeError,
                                        uint32_t* dids_out,
                                        int const nchannels,
                                        double* weights,
                                        uint32_t* flags) {
      constexpr int nsamples = EcalDataFrame_Ph2::MAXSAMPLES;
      int const tx = threadIdx.x + blockIdx.x * blockDim.x;
      unsigned int nchannels_per_block = blockDim.x;

      if (tx < nchannels) {
        auto const did = DetId{dids[tx]};
        //dynamic shared memory
        extern __shared__ char shared_mem[];
        double* shr_weights = (double*)&shared_mem[0];
        float* shr_amp = (float*)&shared_mem[nsamples * sizeof(double)];
        uint16_t* shr_digis = (uint16_t*)&shared_mem[nsamples * sizeof(double)+ nchannels_per_block * sizeof(float)];

        shr_weights = weights;

        unsigned int bx = blockIdx.x;  //block index
        unsigned int btx = threadIdx.x;

        for (int sample = 0; sample < nsamples; ++sample) {
          const unsigned int idx = threadIdx.x * nsamples + sample;
          shr_digis[idx] = digis_in[bx * nchannels_per_block * nsamples + idx];
        }

        shr_amp[btx] = 0.0;
        CMS_UNROLL_LOOP
        for (int sample = 0; sample < nsamples; ++sample) {
          const unsigned int idx = threadIdx.x * nsamples + sample;
          const auto shr_digi = shr_digis[idx];
          shr_amp[btx] += (static_cast<float>(ecalLiteDTU::adc(shr_digi)) * ecalPh2::gains[ecalLiteDTU::gainId(shr_digi)] *
                           shr_weights[sample]);
        }
        amplitude[tx] = shr_amp[btx];
        amplitudeError[tx] = 1.0f;
        dids_out[tx] = did.rawId();
        flags[tx] = 0;
        if (ecalLiteDTU::gainId(shr_digis[btx * nsamples + nsamples - 1])) {
          flags[tx] = 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;
        }

      }  //if within nchannels
    }    //kernel
  }      //namespace weights
}  //namespace ecal