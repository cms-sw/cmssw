#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/Pythia6Interface/interface/PythiaSource.h"
#include "GeneratorInterface/Pythia6Interface/interface/PythiaProducer.h"
#include "GeneratorInterface/Pythia6Interface/interface/HepMCProductAnalyzer.h"

  using edm::PythiaSource;
  using edm::PythiaProducer;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(PythiaSource);
  DEFINE_ANOTHER_FWK_MODULE(PythiaProducer);
  DEFINE_ANOTHER_FWK_MODULE(HepMCProductAnalyzer);

