#
# Constants taken from DataFormats/HLTReco/interface/TriggerTypeDefs.h
# for better readability of the python configuration code here
#
# comments are also copied from there
#
# not sure whether this should be a _cfi.py or _cff.py

#----------------------------------------
# enum start value shifted to 81 so as to avoid clashes with PDG codes
# 
# L1 - using cases as defined in enum L1GtObject, file:
# DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h

TriggerL1Mu           = -81
TriggerL1NoIsoEG      = -82
TriggerL1IsoEG        = -83
TriggerL1CenJet       = -84
TriggerL1ForJet       = -85
TriggerL1TauJet       = -86
TriggerL1ETM          = -87
TriggerL1ETT          = -88
TriggerL1HTT          = -89
TriggerL1HTM          = -90
TriggerL1JetCounts    = -91
TriggerL1HfBitCounts  = -92
TriggerL1HfRingEtSums = -93
TriggerL1TechTrig     = -94
TriggerL1Castor       = -95
TriggerL1BPTX         = -96
TriggerL1GtExternal   = -97

# HLT

TriggerPhoton         = +81
TriggerElectron       = +82
TriggerMuon           = +83
TriggerTau            = +84
TriggerJet            = +85
TriggerBJet           = +86
TriggerMET            = +87
TriggerTET            = +88
TriggerTHT            = +89
TriggerMHT            = +90
TriggerTrack          = +91
TriggerCluster        = +92
TriggerMETSig         = +93
TriggerELongit        = +94
TriggerMHTSig         = +95
TriggerHLongit        = +96
