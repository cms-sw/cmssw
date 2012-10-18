#ifndef REGGEGRIBOVPARTONMCGENERATORFILTER_H
#define REGGEGRIBOVPARTONMCGENERATORFILTER_H

#include "GeneratorInterface/ReggeGribovPartonMCInterface/interface/ReggeGribovPartonMCHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace gen
{
   typedef edm::GeneratorFilter<gen::ReggeGribovPartonMCHadronizer, gen::ExternalDecayDriver> ReggeGribovPartonMCGeneratorFilter;
}

#endif //#ifndef REGGEGRIBOVPARTONMCGENERATORFILTER_H
