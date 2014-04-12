# Generate dummy L1TriggerKey with "SubsystemKeysOnly" label
from CondTools.L1Trigger.L1TriggerKeyDummy_cff import *
L1TriggerKeyDummy.objectKeys = cms.VPSet()
L1TriggerKeyDummy.label = "SubsystemKeysOnly"

# Generate object keys for each subsystem
from L1TriggerConfig.DTTrackFinder.L1DTTFRSKeysOnline_cfi import *

from L1TriggerConfig.RCTConfigProducers.RCT_RSKeysOnline_cfi import *

from L1TriggerConfig.GctConfigProducers.L1GctRSObjectKeysOnline_cfi import *
L1GctRSObjectKeysOnline.subsystemLabel = "GCT"

from L1TriggerConfig.GMTConfigProducers.L1MuGMTRSKeysOnline_cfi import *

from L1TriggerConfig.L1GtConfigProducers.l1GtRsObjectKeysOnline_cfi import *
#l1GtRsObjectKeysOnline.EnableL1GtPrescaleFactorsAlgoTrig = False
#l1GtRsObjectKeysOnline.EnableL1GtPrescaleFactorsTechTrig = False
#l1GtRsObjectKeysOnline.EnableL1GtTriggerMaskAlgoTrig = True
#l1GtRsObjectKeysOnline.EnableL1GtTriggerMaskTechTrig = False
#l1GtRsObjectKeysOnline.EnableL1GtTriggerMaskVetoTechTrig = False
l1GtRsObjectKeysOnline.subsystemLabel = "GT"

# Collate subsystem object keys
from CondTools.L1Trigger.L1TriggerKeyOnline_cfi import *
L1TriggerKeyOnline.subsystemLabels = cms.vstring( 'DTTF',
                                                  'RCT_',
                                                  'GCT',
                                                  'L1MuGMT',
                                                  'GT' )
