import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
dimuonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Add this to access 8E29 menu
#dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
# for 8E29 menu
#dimuonsHLTFilter.HLTPaths = ["HLT_Mu3", "HLT_DoubleMu3"]
# for 1E31 menu
#dimuonsHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]
dimuonsHLTFilter.HLTPaths = ["HLT_Mu9"]
