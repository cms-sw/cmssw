import FWCore.ParameterSet.Config as cms

import  RecoMET.METFilters.BadPFMuonFilter_cfi as _mod

BadPFMuonSummer16Filter = _mod.BadPFMuonFilter.clone(
  filterType = "BadPFMuonSummer16",
  innerTrackRelErr = 0.5,
)
