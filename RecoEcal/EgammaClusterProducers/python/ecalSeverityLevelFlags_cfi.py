import FWCore.ParameterSet.Config as cms

ecalSeverityLevelFlag_kGood = 0                  # rechit is ok
ecalSeverityLevelFlag_kProblematic = 1           # problematic
ecalSeverityLevelFlag_kRecovedered = 2           # recovedered
ecalSeverityLevelFlag_kTime = 3                  # time is wrong (e.g. early spike)
ecalSeverityLevelFlag_kWeird = 4                 # weird (e.g. spikeId)
ecalSeverityLevelFlag_kBad = 5                   # bad
