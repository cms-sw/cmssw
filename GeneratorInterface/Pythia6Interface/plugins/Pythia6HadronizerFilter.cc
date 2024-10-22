// -*- C++ -*-

#include "GeneratorInterface/Core/interface/HadronizerFilter.h"
#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"
#include "Pythia6Hadronizer.h"

namespace gen {
  typedef edm::HadronizerFilter<gen::Pythia6Hadronizer, gen::ExternalDecayDriver> Pythia6HadronizerFilter;
}

using gen::Pythia6HadronizerFilter;
DEFINE_FWK_MODULE(Pythia6HadronizerFilter);
