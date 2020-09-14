import FWCore.ParameterSet.Config as cms

import  RecoMET.METFilters.badParticleFilter_cfi  as _mod

BadChargedCandidateFilter = _mod.badParticleFilter.clone(
    filterType  ="BadChargedCandidate",
    maxDR = 0.00001,
    minPtDiffRel = 0.00001
)
