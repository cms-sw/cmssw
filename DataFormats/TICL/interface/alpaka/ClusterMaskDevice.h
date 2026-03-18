
#pragma once

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/TICL/interface/ClusterMask.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using ClusterMaskDevice = PortableCollection<Device, ::ticl::ClusterMask>

}
