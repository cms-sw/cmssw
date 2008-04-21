import FWCore.ParameterSet.Config as cms

#Include configuration ParameterSets
from L1Trigger.DTTrigger.dttpg_conf_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigMap_cff import *
L1DTConfig = cms.ESProducer("DTConfigTrivialProducer",
    DTTPGMapBlock,
    DTTPGParametersBlock
)


