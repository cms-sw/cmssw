#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/PFCandProducer/interface/PdgIdPFCandidateSelectorDefinition.h"

typedef ObjectSelector<PdgIdPFCandidateSelectorDefinition> PdgIdPFCandidateSelector;

DEFINE_FWK_MODULE(PdgIdPFCandidateSelector);
