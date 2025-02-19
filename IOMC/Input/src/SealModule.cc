#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOMC/Input/interface/MCFileSource.h"


  using edm::MCFileSource; 
  
  DEFINE_FWK_INPUT_SOURCE(MCFileSource);

