
#pragma once

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  GENERATE_SOA_LAYOUT(ClusterMaskLayout, SOA_COLUMN(float, mask));

  using ClusterMask = ClusterMaskLayout<>;
  using ClusterMaskView = ClusterMask::View;
  using ClusterMaskConstView = ClusterMask::ConstView;

}  // namespace ticl
