import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToTauTauETauHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToTauTauETauHLTFilter.HLTPaths = ['HLT1Electron', 'HLTXElectronTau']

