import FWCore.ParameterSet.Config as cms

import  RecoMET.METFilters.badParticleFilter_cfi as _mod

BadPFMuonSummer16Filter = _mod.badParticleFilter.clone(
  filterType = "BadPFMuonSummer16",
  innerTrackRelErr = 0.5,
)
