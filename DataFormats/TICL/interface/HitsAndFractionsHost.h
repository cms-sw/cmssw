
#pragma once

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/AssociationMap.h"
#include "DataFormats/TICL/interface/HitAndFraction.h"

namespace ticl {

  using HitsAndFractionsHost = PortableHostCollection<ticl::AssociationMap<int, HitAndFraction>>;

}
