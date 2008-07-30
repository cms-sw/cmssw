import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToMuMuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToMuMuHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso']

