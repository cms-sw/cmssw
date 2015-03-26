#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "RecoTauTag/TauTagTools/plugins/PFTauSelectorDefinition.h"

typedef ObjectSelectorStream<PFTauSelectorDefinition> PFTauSelector;

DEFINE_FWK_MODULE(PFTauSelector);
