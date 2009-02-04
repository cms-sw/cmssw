#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoParticleFlow/PFClusterShapeProducer/interface/PFClusterShapeProducer.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFClusterShapeProducer);
