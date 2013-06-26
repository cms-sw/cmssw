#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "JetMETCorrections/MCJet/plugins/CaloMCTruthTreeProducer.h"
#include "JetMETCorrections/MCJet/plugins/PFMCTruthTreeProducer.h"


DEFINE_FWK_MODULE(CaloMCTruthTreeProducer);
DEFINE_FWK_MODULE(PFMCTruthTreeProducer);
