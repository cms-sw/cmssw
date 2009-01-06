// -*- C++ -*-

#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "Pythia6Hadronizer.h"

namespace gen
{
  typedef edm::HadronizerFilter<gen::Pythia6Hadronizer> Pythia6HadronizerFilter;
}

using gen::Pythia6HadronizerFilter;
DEFINE_FWK_MODULE(Pythia6HadronizerFilter);

