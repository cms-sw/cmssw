// CepGen-CMSSW interfacing module
//   2022-2024, Laurent Forthomme

#ifndef GeneratorInterface_CepGenInterface_CepGenGeneratorFilter_h
#define GeneratorInterface_CepGenInterface_CepGenGeneratorFilter_h

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/CepGenInterface/interface/CepGenEventGenerator.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace edm {
  template <>
  inline GeneratorFilter<gen::CepGenEventGenerator, gen::ExternalDecayDriver>::GeneratorFilter(
      const ParameterSet& iConfig)
      : hadronizer_(iConfig, consumesCollector()) {
    init(iConfig);
  }
}  // namespace edm

namespace gen {
  using CepGenGeneratorFilter = edm::GeneratorFilter<CepGenEventGenerator, gen::ExternalDecayDriver>;
}

#endif
