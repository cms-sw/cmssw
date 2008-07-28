#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MCTruthTreeProducer.h"
#include "PFJetsCorExample.h"
#include "CaloJetsCorExample.h"

using cms::MCTruthTreeProducer;
using cms::PFJetsCorExample;
using cms::CaloJetsCorExample;

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MCTruthTreeProducer);
DEFINE_ANOTHER_FWK_MODULE(PFJetsCorExample);
DEFINE_ANOTHER_FWK_MODULE(CaloJetsCorExample);
