#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"
#include "Cascade2Hadronizer.h"

namespace gen {
  typedef edm::GeneratorFilter<gen::Cascade2Hadronizer, gen::ExternalDecayDriver> Cascade2GeneratorFilter;
}

using gen::Cascade2GeneratorFilter;
DEFINE_FWK_MODULE(Cascade2GeneratorFilter);
