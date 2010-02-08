#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/PFCandProducer/plugins/PtMinPFCandidateSelectorDefinition.h"

typedef ObjectSelector<PtMinPFCandidateSelectorDefinition> PtMinPFCandidateSelector;

DEFINE_ANOTHER_FWK_MODULE(PtMinPFCandidateSelector);

