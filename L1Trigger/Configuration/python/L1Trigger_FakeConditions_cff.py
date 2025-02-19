import FWCore.ParameterSet.Config as cms

# Dummy records for L1TriggerKey and L1TriggerKeyList
from CondTools.L1Trigger.L1TriggerKeyRcdSource_cfi import *
from CondTools.L1Trigger.L1TriggerKeyListRcdSource_cfi import *

# L1 Calo configuration
from L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff import *
from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomConfig_cff import *

# L1 Muon configuration
from L1TriggerConfig.DTTrackFinder.L1DTTrackFinderConfig_cff import *
from L1TriggerConfig.CSCTFConfigProducers.L1CSCTFConfig_cff import *
from L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff import *
from L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff import *
from L1TriggerConfig.RPCTriggerConfig.RPCHsbConfig_cff import *
from L1TriggerConfig.RPCTriggerConfig.RPCBxOrConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *

# GT configuration
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *

# these are never stored in L1 DB
# they should be moved to CSC/DT/RPC fake conditions
#from L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff import *
#from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff import *
#from L1TriggerConfig.RPCTriggerConfig.RPCConeConfig_cff import *
#from L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff import *
