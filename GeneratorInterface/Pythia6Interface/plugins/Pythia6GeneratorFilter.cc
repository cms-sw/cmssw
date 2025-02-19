// -*- C++ -*-

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"
#include "Pythia6Hadronizer.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::Pythia6Hadronizer, gen::ExternalDecayDriver> Pythia6GeneratorFilter;
}

using gen::Pythia6GeneratorFilter;
DEFINE_FWK_MODULE(Pythia6GeneratorFilter);

