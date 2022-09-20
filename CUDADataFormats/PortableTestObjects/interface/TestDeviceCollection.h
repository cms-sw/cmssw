#ifndef CUDADataFormats_PortableTestObjects_interface_TestDeviceCollection_h
#define CUDADataFormats_PortableTestObjects_interface_TestDeviceCollection_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

namespace cudatest {

  // SoA with x, y, z, id fields in device global memory
  using TestDeviceCollection = cms::cuda::PortableDeviceCollection<portabletest::TestSoA>;

}  // namespace cudatest

#endif  // CUDADataFormats_PortableTestObjects_interface_TestDeviceCollection_h
