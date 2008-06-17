import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
zToTauTau_DoubleTauHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
zToTauTau_DoubleTauHLTFilter.HLTPaths = ['HLT_DoubleIsoTau_Trk3']

