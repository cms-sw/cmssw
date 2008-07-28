#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MCTruthTreeProducer.h"

using cms::MCTruthTreeProducer;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MCTruthTreeProducer);
