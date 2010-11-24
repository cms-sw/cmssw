#ifndef gen_PyquenGeneratorFilter_h
#define gen_PyquenGeneratorFilter_h

#include "GeneratorInterface/PyquenInterface/interface/PyquenHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"

namespace gen
{
   typedef edm::GeneratorFilter<gen::PyquenHadronizer> PyquenGeneratorFilter;
}

#endif
