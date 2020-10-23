#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalPedestalsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalPedestalsGPU_h

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalPedestalsGPU {
public:
  struct Product {
    ~Product();
    float *values;
    float *widths;
    /*
        float *value0;
        float *value1;
        float *value2;
        float *value3;
        float *width0;
        float *width1;
        float *width2;
        float *width3;
        */
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalPedestalsGPU(HcalPedestals const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalPedestalsGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

  // as in cpu version
  bool unitIsADC() const { return unitIsADC_; }

  uint32_t offsetForHashes() const { return offsetForHashes_; }

private:
  bool unitIsADC_;
  uint64_t totalChannels_;
  uint32_t offsetForHashes_;
  std::vector<float, cms::cuda::HostAllocator<float>> values_, widths_;
  /*
    std::vector<float, cms::cuda::HostAllocator<float>> value0_, value1_, value2_, value3_,
                                                 width0_, width1_, width2_, width3_;
                                                 */

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
