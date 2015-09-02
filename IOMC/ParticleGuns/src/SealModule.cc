#include "FWCore/Framework/interface/MakerMacros.h"
// #include "IOMC/Input/interface/MCFileSource.h"

// Julia Yarba : related to particle gun prototypes
//
//#include "IOMC/ParticleGuns/interface/FlatEGunASCIIWriter.h"

#include "IOMC/ParticleGuns/interface/FileRandomKEThetaGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomEThetaGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomPtThetaGunProducer.h"
#include "IOMC/ParticleGuns/interface/FlatRandomOneOverPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/ExpoRandomPtGunProducer.h"
#include "IOMC/ParticleGuns/interface/ExpoRandomPGunProducer.h"
#include "IOMC/ParticleGuns/interface/MultiParticleInConeGunProducer.h"

// particle gun prototypes
//
  
  
/*
using edm::FlatEGunASCIIWriter;
DEFINE_FWK_MODULE(FlatEGunASCIIWriter);
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
using edm::ExpoRandomPGunProducer;
DEFINE_FWK_MODULE(ExpoRandomPGunProducer);
using edm::MultiParticleInConeGunProducer;
DEFINE_FWK_MODULE(MultiParticleInConeGunProducer);
