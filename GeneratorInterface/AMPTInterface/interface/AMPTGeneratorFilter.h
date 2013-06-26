#ifndef gen_AMPTGeneratorFilter_h
#define gen_AMPTGeneratorFilter_h

#include "GeneratorInterface/AMPTInterface/interface/AMPTHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace gen
{
   typedef edm::GeneratorFilter<gen::AMPTHadronizer, gen::ExternalDecayDriver> AMPTGeneratorFilter;
}

#endif
