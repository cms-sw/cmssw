import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
dimuonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
dimuonsHLTFilter.HLTPaths = ["HLT_Mu3", "HLT_DoubleMu3"]
