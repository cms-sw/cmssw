
#pragma once

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TICL/interface/AssociationMap.h"
#include "DataFormats/TICL/interface/HitAndFraction.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using HitsAndFractionsDevice = PortableCollection<ticl::TICLAssociationMap<int, HitAndFraction>>;

}
