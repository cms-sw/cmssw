#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"

typedef Merger<reco::IsolatedPFCandidateCollection> IsolatedPFCandidateMerger;

DEFINE_ANOTHER_FWK_MODULE(IsolatedPFCandidateMerger);
