#ifndef gen_HydjetGeneratorFilter_h
#define gen_HydjetGeneratorFilter_h

#include "GeneratorInterface/Hydjet2Interface/interface/Hydjet2Hadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::Hydjet2Hadronizer, gen::ExternalDecayDriver> Hydjet2GeneratorFilter;
}

#endif
