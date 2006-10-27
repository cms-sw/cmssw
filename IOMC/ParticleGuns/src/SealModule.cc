#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "IOMC/Input/interface/MCFileSource.h"

// Julia Yarba : related to particle gun prototypes
//
#include "IOMC/ParticleGuns/interface/FlatEGunASCIIWriter.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEGunSource.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtGunSource.h"

/*
  using edm::MCFileSource; 
  DEFINE_SEAL_MODULE();
  DEFINE_ANOTHER_FWK_INPUT_SOURCE(MCFileSource);
*/
// particle gun prototypes
//
  DEFINE_SEAL_MODULE();
  
using edm::FlatEGunASCIIWriter;
DEFINE_ANOTHER_FWK_MODULE(FlatEGunASCIIWriter);
using edm::FlatRandomEGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(FlatRandomEGunSource);
using edm::FlatRandomPtGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(FlatRandomPtGunSource);
