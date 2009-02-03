import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Common.FourVectorHLTriggerOffline_cfi import *

#from L1Trigger.Configuration.L1Config_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_Unprescaled_cff import *

HLTFourVector = cms.Sequence(hltResults)
