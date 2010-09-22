#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/PomwigInterface/interface/PomwigHadronizer.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::PomwigHadronizer> PomwigGeneratorFilter;
}

using gen::PomwigGeneratorFilter;
DEFINE_FWK_MODULE(PomwigGeneratorFilter);
