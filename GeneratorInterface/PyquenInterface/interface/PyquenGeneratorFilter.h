#ifndef gen_PyquenGeneratorFilter_h
#define gen_PyquenGeneratorFilter_h

#include "GeneratorInterface/PyquenInterface/interface/PyquenHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace edm {
  template <>
  inline GeneratorFilter<gen::PyquenHadronizer, gen::ExternalDecayDriver>::GeneratorFilter(ParameterSet const& ps)
      : hadronizer_(ps, consumesCollector()) {
    init(ps);
  }
}  // namespace edm

namespace gen {
  typedef edm::GeneratorFilter<gen::PyquenHadronizer, gen::ExternalDecayDriver> PyquenGeneratorFilter;
}

#endif
