#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "RecoTauTag/TauTagTools/plugins/PFTauSelectorDefinition.h"

typedef ObjectSelector<PFTauSelectorDefinition> PFTauSelector;

DEFINE_FWK_MODULE(PFTauSelector);

