#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/ParticleFlow/interface/PtMinPFCandidateSelectorDefinition.h"


typedef ObjectSelector<pf2pat::PtMinPFCandidateSelectorDefinition> PtMinPFCandidateSelector;

DEFINE_FWK_MODULE(PtMinPFCandidateSelector);

