// -*- C++ -*-

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "Pythia6Hadronizer.h"

namespace gen
{
  typedef edm::GeneratorFilter<gen::Pythia6Hadronizer> Pythia6GeneratorFilter;
}

using gen::Pythia6GeneratorFilter;
DEFINE_FWK_MODULE(Pythia6GeneratorFilter);

