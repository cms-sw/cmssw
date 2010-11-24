#ifndef gen_HijingGeneratorFilter_h
#define gen_HijingGeneratorFilter_h

#include "GeneratorInterface/HijingInterface/interface/HijingHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"

namespace gen
{
   typedef edm::GeneratorFilter<gen::HijingHadronizer> HijingGeneratorFilter;
}

#endif
