# Generate dummy L1TriggerKeyExt with "SubsystemKeysOnly" label
from CondTools.L1TriggerExt.L1TriggerKeyDummyExt_cff import *
L1TriggerKeyDummyExt.objectKeys = cms.VPSet()
L1TriggerKeyDummyExt.label = "SubsystemKeysOnly"

# Generate object keys for each subsystem
##from L1TriggerConfig.DTTrackFinder.L1DTTFRSKeysOnline_cfi import *

# Collate subsystem object keys
from CondTools.L1TriggerExt.L1TriggerKeyOnlineExt_cfi import *
L1TriggerKeyOnlineExt.subsystemLabels = cms.vstring( ) #'DTTF' )
