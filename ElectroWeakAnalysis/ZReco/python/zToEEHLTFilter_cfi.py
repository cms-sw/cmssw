import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToEEHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToEEHLTFilter.HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R', 'HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R', 'HLT_IsoEle8_IsoMu7']

