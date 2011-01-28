#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

typedef Merger<reco::IsolatedPFCandidateCollection> IsolatedPFCandidateMerger;

DEFINE_FWK_MODULE(IsolatedPFCandidateMerger);
