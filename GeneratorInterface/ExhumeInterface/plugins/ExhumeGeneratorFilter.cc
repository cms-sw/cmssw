#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExhumeInterface/interface/ExhumeHadronizer.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::ExhumeHadronizer> ExhumeGeneratorFilter;
}

using gen::ExhumeGeneratorFilter;
DEFINE_FWK_MODULE(ExhumeGeneratorFilter);
