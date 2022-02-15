#include <cstdlib>
#include <limits>

#include <cuda.h>

#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"


#include "EcalUncalibRecHitPhase2WeightsKernels.h"
#include "KernelHelpers.h"

#include "EigenMatrixTypes_gpu.h"

#include "DeclsForKernelsPh2WeightsGPU.h"

// Kernel which executes weights algorithm on device

namespace ecal
{
  namespace weights
  {

    __global__ void Phase2WeightsKernel(uint16_t const* digis_in_eb,
                          uint32_t const* dids_eb,
                          ::ecal::reco::StorageScalarType* amplitudeEB,
                          uint32_t* dids_outEB,
                          int const nchannels,
                          double* weights_d,
                          uint32_t* flagsEB
                          // ,double* debug_d
                          )
    {

    constexpr int nsamples = EcalDataFrame_Ph2::MAXSAMPLES;
    int const tx = threadIdx.x + blockIdx.x * blockDim.x;
    auto const* digis_in = digis_in_eb;
    auto const* dids = dids_eb;

if (tx < nchannels){
    
    auto const did = DetId{dids[tx]};

    double amp = 0.0;
    bool g1 = false;

    float gains[2] = {10., 1.}; //since ecalPh2::gains doesn't work
    double gratio = 0.0;

    // CMS_UNROLL_LOOP
    for(int sample = 0; sample < nsamples; ++sample) 
    {
    double adc = 1.0 * ecalLiteDTU::adc(digis_in[tx * nsamples + sample]);
    int gainId = ecalLiteDTU::gainId(digis_in[tx * nsamples + sample]);  // is the gain Id added properly to the digis in the first place?
    // gratio = ecalPh2::gains[gainId];   this gives error undefined in device code, hence it is hard coded above
    gratio = gains[gainId];
    if (gainId == 1)
      {g1= true;}
      amp = amp + (adc * gratio * weights_d[sample]); //weights_d might not have been copied properly?

    }
    
     // debugging============================
    // if(tx < 16){
    //   debug_d[tx] = amp;
    // }
    // //

    amplitudeEB[tx] = amp;
    // chi2EB = 0.;
    // g_pedestalEB = 0.;
    dids_outEB[tx] = did.rawId();
    flagsEB = 0;
    if (g1)
      {
        // flagsEB[inputCh] |= 0x1 << EcalUncalibratedRecHit::kHasSwitchToGain1;
        flagsEB[tx] = EcalUncalibratedRecHit::kHasSwitchToGain1;
      }

    }
  
    }


  }
}