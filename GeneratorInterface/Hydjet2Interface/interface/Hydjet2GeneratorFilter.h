#ifndef gen_Hydjet2GeneratorFilter_h
#define gen_Hydjet2GeneratorFilter_h

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"
#include "GeneratorInterface/Hydjet2Interface/interface/Hydjet2Hadronizer.h"

namespace edm {
  template <>
  inline GeneratorFilter<gen::Hydjet2Hadronizer, gen::ExternalDecayDriver>::GeneratorFilter(ParameterSet const &ps)
      : hadronizer_(ps, consumesCollector()) {
    init(ps);
  }
}  // namespace edm

namespace gen {
  typedef edm::GeneratorFilter<gen::Hydjet2Hadronizer, gen::ExternalDecayDriver> Hydjet2GeneratorFilter;
}

#endif
