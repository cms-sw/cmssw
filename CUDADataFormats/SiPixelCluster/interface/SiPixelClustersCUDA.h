#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

#include <cuda_runtime.h>

GENERATE_SOA_LAYOUT(SiPixelClustersCUDALayout,
                    SOA_COLUMN(uint32_t, moduleStart),
                    SOA_COLUMN(uint32_t, clusInModule),
                    SOA_COLUMN(uint32_t, moduleId),
                    SOA_COLUMN(uint32_t, clusModuleStart))

using SiPixelClustersCUDASoA = SiPixelClustersCUDALayout<>;
using SiPixelClustersCUDASOAView = SiPixelClustersCUDALayout<>::View;
using SiPixelClustersCUDASOAConstView = SiPixelClustersCUDALayout<>::ConstView;

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition, i.e.,
// by adding a public class attribute like:
// cms::cuda::Portabledevicecollection<SiPixelClustersCUDALayout<>> collection;
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelClustersCUDA : public cms::cuda::PortableDeviceCollection<SiPixelClustersCUDALayout<>> {
public:
  SiPixelClustersCUDA() = default;
  ~SiPixelClustersCUDA() = default;

  explicit SiPixelClustersCUDA(size_t maxModules, cudaStream_t stream)
      : PortableDeviceCollection<SiPixelClustersCUDALayout<>>(maxModules + 1, stream) {}

  SiPixelClustersCUDA(SiPixelClustersCUDA &&) = default;
  SiPixelClustersCUDA &operator=(SiPixelClustersCUDA &&) = default;

  void setNClusters(uint32_t nClusters, int32_t offsetBPIX2) {
    nClusters_h = nClusters;
    offsetBPIX2_h = offsetBPIX2;
  }

  uint32_t nClusters() const { return nClusters_h; }
  int32_t offsetBPIX2() const { return offsetBPIX2_h; }

private:
  uint32_t nClusters_h = 0;
  int32_t offsetBPIX2_h = 0;
};

#endif  // CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
