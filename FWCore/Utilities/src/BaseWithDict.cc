
#include "Reflex/Type.h"

#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace edm {
  TypeWithDict
  BaseWithDict::toType() const {
    return (TypeWithDict(base_.ToType()));
  }
}
