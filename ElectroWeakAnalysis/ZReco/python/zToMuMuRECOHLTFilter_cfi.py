import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToMuMuRECOHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToMuMuRECOHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_Mu15_L1Mu7', 'HLT_DoubleMu3', 'HLT_DoubleMu7_Z']

