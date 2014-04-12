import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
electronDQMIsoDistTrigger = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
electronDQMIsoDistTrigger.HLTPaths = ['HLT_LooseIsoEle15_LW_L1R']

