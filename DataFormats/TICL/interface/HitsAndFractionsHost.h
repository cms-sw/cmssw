
#pragma once

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/AssociationMap.h"
#include "DataFormats/TICL/interface/HitAndFraction.h"

namespace ticl {

  using TICLAssociationMap_t = AssociationMapLayout<int, HitAndFraction>::Layout<128, false>;
  using HitsAndFractionsHost = PortableHostCollection<TICLAssociationMap_t>;

}  // namespace ticl
