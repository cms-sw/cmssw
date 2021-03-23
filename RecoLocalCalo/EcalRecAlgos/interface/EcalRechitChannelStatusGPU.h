#ifndef RecoLocalCalo_EcalRecAlgos_interface_EcalRechitChannelStatusGPU_h
#define RecoLocalCalo_EcalRecAlgos_interface_EcalRechitChannelStatusGPU_h

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

class EcalRechitChannelStatusGPU {
public:
  struct Product {
    ~Product();
    uint16_t* status = nullptr;
  };

#ifndef __CUDACC__

  //
  EcalRechitChannelStatusGPU(EcalChannelStatus const&);

  // will call dealloation for Product thru ~Product
  ~EcalRechitChannelStatusGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //
  static std::string name() { return std::string{"ecalRechitChannelStatusGPU"}; }

private:
  // in the future, we need to arrange so to avoid this copy on the host
  // store eb first then ee
  std::vector<uint16_t, cms::cuda::HostAllocator<uint16_t>> status_;

  cms::cuda::ESProduct<Product> product_;

#endif  // __CUDACC__
};

#endif  // RecoLocalCalo_EcalRecAlgos_interface_EcalRechitChannelStatusGPU_h
