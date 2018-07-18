#ifndef FWCore_ParameterSet_ParameterSetTraits_h
#define FWCore_ParameterSet_ParameterSetTraits_h

#include "FWCore/Utilities/interface/value_ptr.h"

namespace edm {

  class ParameterSetDescription;

  template <>
  struct value_ptr_traits<ParameterSetDescription> {
    static ParameterSetDescription* clone(ParameterSetDescription const* p);
    static void destroy(ParameterSetDescription* p);
  };
}
#endif
