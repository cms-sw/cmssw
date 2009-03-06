import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
dimuonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
dimuonsHLTFilter.HLTPaths = ['HLT1MuonPrescalePt3', 'HLT1MuonPrescalePt7x7', 'HLT1MuonIso', 'HLT1MuonNonIso15', 'HLT2MuonNonIso']

