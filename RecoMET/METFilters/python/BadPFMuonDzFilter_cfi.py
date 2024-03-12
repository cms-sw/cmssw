import FWCore.ParameterSet.Config as cms

import  RecoMET.METFilters.badParticleFilter_cfi as _mod

BadPFMuonDzFilter = _mod.badParticleFilter.clone(
  filterType = "BadPFMuonDz",
  minDzBestTrack = 0.5
)
# foo bar baz
# eD9cWp3MJjRAs
# mwY0BGfBu2wfI
