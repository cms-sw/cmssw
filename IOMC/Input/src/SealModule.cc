#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOMC/Input/interface/MCFileSource.h"


  using edm::MCFileSource; 
  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(MCFileSource);

