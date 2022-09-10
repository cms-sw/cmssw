#ifndef CUDADataFormats_PortableTestObjects_interface_TestHostCollection_h
#define CUDADataFormats_PortableTestObjects_interface_TestHostCollection_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"

namespace cudatest {

  // SoA with x, y, z, id fields in host memory
  using TestHostCollection = cms::cuda::PortableHostCollection<portabletest::TestSoA>;

}  // namespace cudatest

#endif  // CUDADataFormats_PortableTestObjects_interface_TestHostCollection_h
