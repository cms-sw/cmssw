#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalRecoParamsWithPulseShapesGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalRecoParamsWithPulseShapesGPU_h

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class HcalRecoParams;

//
// TODO: HcalPulseShapes will need to be used via ESSource
// This is a workaround: precompute/store/transfer what's needed only
//
class HcalRecoParamsWithPulseShapesGPU {
public:
  struct Product {
    ~Product();
    uint32_t *param1 = nullptr, *param2 = nullptr;
    uint32_t *ids = nullptr;

    // These guys come directly from PulseShapeFunctor class
    float *acc25nsVec = nullptr, *diff25nsItvlVec = nullptr, *accVarLenIdxMinusOneVec = nullptr,
          *diffVarItvlIdxMinusOneVec = nullptr, *accVarLenIdxZEROVec = nullptr, *diffVarItvlIdxZEROVec = nullptr;
  };

#ifndef __CUDACC__
  // rearrange reco params
  HcalRecoParamsWithPulseShapesGPU(HcalRecoParams const &);

  // will trigger deallocation of Product thru ~Product
  ~HcalRecoParamsWithPulseShapesGPU() = default;

  // get device pointers
  Product const &getProduct(cudaStream_t) const;

private:
  uint64_t totalChannels_;  // hb + he
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> param1_;
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> param2_;
  std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> ids_;

  std::vector<float, cms::cuda::HostAllocator<float>> acc25nsVec_;                 // 256
  std::vector<float, cms::cuda::HostAllocator<float>> diff25nsItvlVec_;            // 256
  std::vector<float, cms::cuda::HostAllocator<float>> accVarLenIdxMinusOneVec_;    // 25
  std::vector<float, cms::cuda::HostAllocator<float>> diffVarItvlIdxMinusOneVec_;  // 25
  std::vector<float, cms::cuda::HostAllocator<float>> accVarLenIdxZEROVec_;        // 25
  std::vector<float, cms::cuda::HostAllocator<float>> diffVarItvlIdxZEROVec_;      // 25

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
