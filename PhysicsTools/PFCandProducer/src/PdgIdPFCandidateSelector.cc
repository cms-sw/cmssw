#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


// include the definition of you selection
// #include "PhysicsTools/PFCandProducer/interface/PFCandidateSelectorDefinition.h"

// define your producer name
// typedef ObjectSelector<PFCandidateSelectorDefinition> PFCandidateSelector;
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PdgIdSelector.h"

typedef SingleObjectSelector<reco::PFCandidateCollection,
			     PdgIdSelector > PdgIdPFCandidateSelector;

DEFINE_ANOTHER_FWK_MODULE(PdgIdPFCandidateSelector);
