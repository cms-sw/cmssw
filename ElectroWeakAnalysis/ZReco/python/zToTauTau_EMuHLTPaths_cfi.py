import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToTauTau_EMuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToTauTau_EMuHLTFilter.HLTPaths = ['HLTXElectronMuon', 'HLT1MuonIso', 'HLT1Electron']

