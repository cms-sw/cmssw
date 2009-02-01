#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "RecoTauTag/TauTagTools/plugins/CaloTauSelectorDefinition.h"

typedef ObjectSelector<CaloTauSelectorDefinition> CaloTauSelector;

DEFINE_FWK_MODULE(CaloTauSelector);

