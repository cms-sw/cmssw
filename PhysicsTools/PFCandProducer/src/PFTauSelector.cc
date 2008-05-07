#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/PFCandProducer/interface/PFTauSelectorDefinition.h"

typedef ObjectSelector<PFTauSelectorDefinition> PFTauSelector;

DEFINE_ANOTHER_FWK_MODULE(PFTauSelector);

