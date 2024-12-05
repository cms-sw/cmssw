#ifndef DataFormats_SiPixelClusterSoA_interface_SiPixelClustersHost_h
#define DataFormats_SiPixelClusterSoA_interface_SiPixelClustersHost_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelClustersHost : public PortableHostCollection<SiPixelClustersSoA> {
public:
  SiPixelClustersHost(edm::Uninitialized) : PortableHostCollection<SiPixelClustersSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelClustersHost(size_t maxModules, TQueue queue)
      : PortableHostCollection<SiPixelClustersSoA>(maxModules + 1, queue) {}

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

#endif  // DataFormats_SiPixelClusterSoA_interface_SiPixelClustersHost_h
