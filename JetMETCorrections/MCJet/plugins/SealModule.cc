#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "JetMETCorrections/MCJet/plugins/MCTruthTreeProducer.h"

using cms::MCTruthTreeProducer;

DEFINE_SEAL_MODULE();
typedef MCTruthTreeProducer<CaloJet> CaloMCTruthTreeProducer;
DEFINE_ANOTHER_FWK_MODULE(CaloMCTruthTreeProducer);

typedef MCTruthTreeProducer<PFJet> PFMCTruthTreeProducer;
DEFINE_ANOTHER_FWK_MODULE(PFMCTruthTreeProducer);
