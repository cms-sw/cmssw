#ifndef FWCore_Sources_VectorInputSourceDescription_h
#define FWCore_Sources_VectorInputSourceDescription_h

/*----------------------------------------------------------------------

VectorInputSourceDescription : the stuff that is needed to configure
a VectorinputSource that does not come in through the ParameterSet  
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Sources/interface/SciTagCategoryForEmbeddedSources.h"

#include <memory>

namespace edm {
  class PreallocationConfiguration;
  class ProductRegistry;

  struct VectorInputSourceDescription {
    VectorInputSourceDescription() : productRegistry_(nullptr) {}

    VectorInputSourceDescription(std::shared_ptr<ProductRegistry> preg,
                                 PreallocationConfiguration const& allocations,
                                 SciTagCategoryForEmbeddedSources cat = SciTagCategoryForEmbeddedSources::Embedded)
        : productRegistry_(preg), allocations_(&allocations), cat_(cat) {}

    std::shared_ptr<ProductRegistry> productRegistry_;
    PreallocationConfiguration const* allocations_;
    SciTagCategoryForEmbeddedSources cat_;
  };
}  // namespace edm

#endif
