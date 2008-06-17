import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToTauTau_MuTauHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToTauTau_MuTauHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_IsoMu14_IsoTau_Trk3']

