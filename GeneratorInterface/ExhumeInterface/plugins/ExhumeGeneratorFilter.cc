#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExhumeInterface/interface/ExhumeHadronizer.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::ExhumeHadronizer, gen::ExternalDecayDriver> ExhumeGeneratorFilter;
}

using gen::ExhumeGeneratorFilter;
DEFINE_FWK_MODULE(ExhumeGeneratorFilter);
