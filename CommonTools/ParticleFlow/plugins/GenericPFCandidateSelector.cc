#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/ParticleFlow/interface/GenericPFCandidateSelectorDefinition.h"

typedef ObjectSelector<pf2pat::GenericPFCandidateSelectorDefinition> GenericPFCandidateSelector;

DEFINE_FWK_MODULE(GenericPFCandidateSelector);
