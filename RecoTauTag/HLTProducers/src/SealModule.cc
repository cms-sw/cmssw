#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTauTag/HLTProducers/interface/IsolatedTauJetsSelector.h"
#include "RecoTauTag/HLTProducers/interface/L2TauJetsProvider.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
#include "RecoTauTag/HLTProducers/interface/L2TauJetMerger.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(IsolatedTauJetsSelector);
DEFINE_ANOTHER_FWK_MODULE(L2TauJetsProvider);
DEFINE_ANOTHER_FWK_MODULE(L2TauJetMerger);
DEFINE_ANOTHER_FWK_MODULE(CaloTowerCreatorForTauHLT);



