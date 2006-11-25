#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "JetMETCorrections/MCJet/interface/MCJetProducer.h"
#include "JetMETCorrections/MCJet/interface/SimJetResponseAnalysis.h"
using cms::MCJet;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MCJet);
DEFINE_ANOTHER_FWK_MODULE(SimJetResponseAnalysis);
