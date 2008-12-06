#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/Pythia8Interface/interface/Pythia8Source.h"
#include "GeneratorInterface/Pythia8Interface/interface/Pythia8Producer.h"

  using edm::Pythia8Source;
  using edm::Pythia8Producer;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(Pythia8Source);
  DEFINE_ANOTHER_FWK_MODULE(Pythia8Producer);

