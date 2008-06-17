import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToTauTauETauHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToTauTauETauHLTFilter.HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_IsoEle12_IsoTau_Trk3']

