#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// #include "IOMC/Input/interface/MCFileSource.h"

// Julia Yarba : related to particle gun prototypes
//
//#include "IOMC/ParticleGuns/interface/FlatEGunASCIIWriter.h"
//#include "IOMC/ParticleGuns/interface/FlatRandomEGunSource.h"
//#include "IOMC/ParticleGuns/interface/FlatRandomPtGunSource.h"
//#include "IOMC/ParticleGuns/interface/FlatRandomEThetaGunSource.h"
//#include "IOMC/ParticleGuns/interface/FlatRandomPtThetaGunSource.h"
//#include "IOMC/ParticleGuns/interface/ExpoRandomPtGunSource.h"
//#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunSource.h"

#include "IOMC/ParticleGuns/interface/FileRandomKEThetaGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEThetaGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtThetaGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomOneOverPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/ExpoRandomPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunProducer.h"

// particle gun prototypes
//
  
  
/*
using edm::FlatEGunASCIIWriter;
DEFINE_FWK_MODULE(FlatEGunASCIIWriter);
using edm::FlatRandomEGunSource;
DEFINE_FWK_INPUT_SOURCE(FlatRandomEGunSource);
using edm::FlatRandomPtGunSource;
DEFINE_FWK_INPUT_SOURCE(FlatRandomPtGunSource);
using edm::FlatRandomEThetaGunSource;
DEFINE_FWK_INPUT_SOURCE(FlatRandomEThetaGunSource);
using edm::FlatRandomPtThetaGunSource;
DEFINE_FWK_INPUT_SOURCE(FlatRandomPtThetaGunSource);
using edm::ExpoRandomPtGunSource;
DEFINE_FWK_INPUT_SOURCE(ExpoRandomPtGunSource);
using edm::MultiParticleInConeGunSource;
DEFINE_FWK_INPUT_SOURCE(MultiParticleInConeGunSource);
*/

using edm::FileRandomKEThetaGunProducer;
DEFINE_FWK_MODULE(FileRandomKEThetaGunProducer);
using edm::FlatRandomEGunProducer;
DEFINE_FWK_MODULE(FlatRandomEGunProducer);
using edm::FlatRandomPtGunProducer;
DEFINE_FWK_MODULE(FlatRandomPtGunProducer);
using edm::FlatRandomEThetaGunProducer;
DEFINE_FWK_MODULE(FlatRandomEThetaGunProducer);
using edm::FlatRandomPtThetaGunProducer;
DEFINE_FWK_MODULE(FlatRandomPtThetaGunProducer);
using edm::FlatRandomOneOverPtGunProducer;
DEFINE_FWK_MODULE(FlatRandomOneOverPtGunProducer);
using edm::ExpoRandomPtGunProducer;
DEFINE_FWK_MODULE(ExpoRandomPtGunProducer);
using edm::MultiParticleInConeGunProducer;
DEFINE_FWK_MODULE(MultiParticleInConeGunProducer);
