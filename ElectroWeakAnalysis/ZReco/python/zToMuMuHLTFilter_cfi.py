import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToMuMuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToMuMuHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_Mu15_L1Mu7', 'HLT_DoubleMu3']

