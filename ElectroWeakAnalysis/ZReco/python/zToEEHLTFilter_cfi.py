import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToEEHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToEEHLTFilter.HLTPaths = ['HLT1Electron', 'HLT1ElectronRelaxed', 'HLT2Electron', 'HLT2ElectronRelaxed', 'HLTXElectronMuon']

