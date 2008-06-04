import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToMuMuRECOHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToMuMuRECOHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso', 'HLT2MuonZ']

