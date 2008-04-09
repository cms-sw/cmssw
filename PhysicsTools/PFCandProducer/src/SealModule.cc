// #include "PluginManager/ModuleDef.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PFCandProducer/interface/PFIsolation.h"
#include "PhysicsTools/PFCandProducer/interface/PFPileUp.h"
#include "PhysicsTools/PFCandProducer/interface/PFTopProjector.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(PFIsolation);
DEFINE_ANOTHER_FWK_MODULE(PFPileUp);
DEFINE_ANOTHER_FWK_MODULE(PFTopProjector);
