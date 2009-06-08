import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Common.FourVectorHLTriggerOffline_cfi import *

#from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
#from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
#from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
#from Configuration.StandardSequences.L1TriggerDefaultMenu_cff import *

HLTFourVector = cms.Sequence(hltriggerResults)
