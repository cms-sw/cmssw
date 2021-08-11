#ifndef gen_HydjetGeneratorFilter_h
#define gen_HydjetGeneratorFilter_h

#include "GeneratorInterface/HydjetInterface/interface/HydjetHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace edm {
  template <>
  inline GeneratorFilter<gen::HydjetHadronizer, gen::ExternalDecayDriver>::GeneratorFilter(ParameterSet const& ps)
      : hadronizer_(ps, consumesCollector()) {
    init(ps);
  }
}  // namespace edm

namespace gen {
  typedef edm::GeneratorFilter<gen::HydjetHadronizer, gen::ExternalDecayDriver> HydjetGeneratorFilter;
}

#endif
