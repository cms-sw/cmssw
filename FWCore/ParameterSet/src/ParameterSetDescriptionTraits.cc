#include "FWCore/ParameterSet/interface/ParameterSetDescriptionTraits.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  ParameterSetDescription* value_ptr_traits<ParameterSetDescription>::clone(ParameterSetDescription const* p) {
    return new ParameterSetDescription(*p);
  }

  void value_ptr_traits<ParameterSetDescription>::destroy(ParameterSetDescription* p) { delete p; }
}  // namespace edm
