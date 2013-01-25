
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TBaseClass.h"

namespace edm {
  BaseWithDict::BaseWithDict() : baseClass_(nullptr) {
  }

  BaseWithDict::BaseWithDict(TBaseClass* baseClass) : baseClass_(baseClass) {
  }

  TypeWithDict
  BaseWithDict::toType() const {
    return TypeWithDict(baseClass_->GetClassPointer(), baseClass_->Property());
  }

  std::string
  BaseWithDict::name() const {
    return baseClass_->GetName();
  }

  bool
  BaseWithDict::isPublic() const {
    return (baseClass_->Property() & kIsPublic);
  }
}
