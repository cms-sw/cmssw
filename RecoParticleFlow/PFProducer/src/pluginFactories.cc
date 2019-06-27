#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
EDM_REGISTER_PLUGINFACTORY(BlockElementImporterFactory, "BlockElementImporterFactory");

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
EDM_REGISTER_PLUGINFACTORY(BlockElementLinkerFactory, "BlockElementLinkerFactory");

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerBase.h"
EDM_REGISTER_PLUGINFACTORY(KDTreeLinkerFactory, "KDTreeLinkerFactory");
