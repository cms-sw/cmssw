import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToTauTau_EMuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToTauTau_EMuHLTFilter.HLTPaths = ['HLT_IsoEle8_IsoMu7', 'HLT_IsoMu11', 'HLT_IsoEle15_L1I']

