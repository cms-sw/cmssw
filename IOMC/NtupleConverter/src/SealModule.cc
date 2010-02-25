#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOMC/NtupleConverter/interface/H2RootNtplSource.h"

  
  using edm::H2RootNtplSource;
  DEFINE_FWK_INPUT_SOURCE(H2RootNtplSource);
