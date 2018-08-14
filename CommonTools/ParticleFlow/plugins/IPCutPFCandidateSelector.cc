#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "CommonTools/ParticleFlow/interface/IPCutPFCandidateSelectorDefinition.h"

typedef ObjectSelectorStream<pf2pat::IPCutPFCandidateSelectorDefinition> IPCutPFCandidateSelector;

DEFINE_FWK_MODULE(IPCutPFCandidateSelector);
