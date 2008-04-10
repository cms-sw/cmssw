#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "IOMC/Input/interface/MCFileSource.h"

// Julia Yarba : related to particle gun prototypes
//
#include "IOMC/ParticleGuns/interface/FlatEGunASCIIWriter.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEGunSource.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtGunSource.h"
#include "IOMC/ParticleGuns/interface/ExpoRandomPtGunSource.h"
#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunSource.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/ExpoRandomPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunProducer.h"

// particle gun prototypes
//
  DEFINE_SEAL_MODULE();
  
using edm::FlatEGunASCIIWriter;
DEFINE_ANOTHER_FWK_MODULE(FlatEGunASCIIWriter);
using edm::FlatRandomEGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(FlatRandomEGunSource);
using edm::FlatRandomPtGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(FlatRandomPtGunSource);
using edm::ExpoRandomPtGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(ExpoRandomPtGunSource);
using edm::MultiParticleInConeGunSource;
DEFINE_ANOTHER_FWK_INPUT_SOURCE(MultiParticleInConeGunSource);
using edm::FlatRandomEGunProducer;
DEFINE_FWK_MODULE(FlatRandomEGunProducer);
using edm::FlatRandomPtGunProducer;
DEFINE_FWK_MODULE(FlatRandomPtGunProducer);
using edm::ExpoRandomPtGunProducer;
DEFINE_FWK_MODULE(ExpoRandomPtGunProducer);
using edm::MultiParticleInConeGunProducer;
DEFINE_FWK_MODULE(MultiParticleInConeGunProducer);
