#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOMC/NtupleConverter/interface/H2RootNtplSource.h"

  DEFINE_SEAL_MODULE();
  using edm::H2RootNtplSource;
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(H2RootNtplSource)
