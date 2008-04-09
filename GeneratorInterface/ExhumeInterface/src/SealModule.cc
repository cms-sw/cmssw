#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/ExhumeInterface/interface/ExhumeSource.h"
#include "GeneratorInterface/ExhumeInterface/interface/ExhumeProducer.h"
 
  using edm::ExhumeSource;
  using edm::ExhumeProducer;

  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(ExhumeSource);
  DEFINE_FWK_MODULE(ExhumeProducer);
