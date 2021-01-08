import FWCore.ParameterSet.Config as cms

import  RecoMET.METFilters.badParticleFilter_cfi as _mod

BadChargedCandidateSummer16Filter = _mod.badParticleFilter.clone(
    filterType  ="BadChargedCandidateSummer16",
    innerTrackRelErr = 0.5,
    minPtDiffRel = -0.5
)
