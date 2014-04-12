#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/PomwigInterface/interface/PomwigHadronizer.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::PomwigHadronizer, gen::ExternalDecayDriver> PomwigGeneratorFilter;
}

using gen::PomwigGeneratorFilter;
DEFINE_FWK_MODULE(PomwigGeneratorFilter);
