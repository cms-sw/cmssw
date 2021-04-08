#ifndef CondFormats_EcalObjects_interface_ElectronicsMappingGPU_h
#define CondFormats_EcalObjects_interface_ElectronicsMappingGPU_h

#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif  // __CUDACC__

namespace ecal {
  namespace raw {

    class ElectronicsMappingGPU {
    public:
      struct Product {
        edm::propagate_const_array<cms::cuda::device::unique_ptr<uint32_t[]>> eid2did;
      };

#ifndef __CUDACC__

      // rearrange pedestals
      ElectronicsMappingGPU(EcalMappingElectronics const&);

      // will call dealloation for Product thru ~Product
      ~ElectronicsMappingGPU() = default;

      // get device pointers
      Product const& getProduct(cudaStream_t) const;

      //
      static std::string name() { return std::string{"ecalElectronicsMappingGPU"}; }

    private:
      // in the future, we need to arrange so to avoid this copy on the host
      // store eb first then ee
      std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> eid2did_;

      cms::cuda::ESProduct<Product> product_;
#endif  // __CUDACC__
    };

  }  // namespace raw
}  // namespace ecal

#endif  // CondFormats_EcalObjects_interface_ElectronicsMappingGPU_h
