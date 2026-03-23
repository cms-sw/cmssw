
#pragma once

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/ClusterMask.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using ClusterMaskDevice = PortableCollection<::ticl::ClusterMask>;

}
