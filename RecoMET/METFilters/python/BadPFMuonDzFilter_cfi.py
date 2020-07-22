import FWCore.ParameterSet.Config as cms

import  RecoMET.METFilters.BadPFMuonFilter_cfi as _mod

BadPFMuonDzFilter = _mod.BadPFMuonFilter.clone(
  filterType = "BadPFMuonDz",
  minDzBestTrack = 0.5
)
