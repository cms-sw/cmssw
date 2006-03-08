#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/MCJet/interface/MCJetProducer.h"
using cms::MCJet;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MCJet)
