#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IOMC/Input/interface/MCFileSource.h"

// Julia Yarba : related to particle gun prototypes
//
#include "IOMC/Input/interface/FlatEGunASCIIWriter.h"
#include "IOMC/Input/interface/FlatRandomEGunSource.h"

  using edm::MCFileSource; 
  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(MCFileSource)

// particle gun prototypes
//
using edm::FlatEGunASCIIWriter;
DEFINE_ANOTHER_FWK_MODULE(FlatEGunASCIIWriter)
using edm::FlatRandomEGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(FlatRandomEGunSource)
