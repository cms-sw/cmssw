#ifndef FWCore_Sources_VectorInputSourceDescription_h
#define FWCore_Sources_VectorInputSourceDescription_h

/*----------------------------------------------------------------------

VectorInputSourceDescription : the stuff that is needed to configure
a VectorinputSource that does not come in through the ParameterSet  
----------------------------------------------------------------------*/

#include "FWCore/Framework/src/PreallocationConfiguration.h"

#include <memory>

namespace edm {
  class PreallocationConfiguration;
  class ProductRegistry;

  struct VectorInputSourceDescription {
    VectorInputSourceDescription() :
      productRegistry_(nullptr) {
    }

    VectorInputSourceDescription(std::shared_ptr<ProductRegistry> preg, PreallocationConfiguration const& allocations) :
      productRegistry_(preg), allocations_(&allocations) {
    }

    std::shared_ptr<ProductRegistry> productRegistry_;
    PreallocationConfiguration const* allocations_;
  };
}

#endif
