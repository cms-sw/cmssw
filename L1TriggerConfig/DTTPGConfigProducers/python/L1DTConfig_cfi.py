import FWCore.ParameterSet.Config as cms

#Include configuration ParameterSets
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigParams_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigMap_cff import *
L1DTConfig = cms.ESProducer("DTConfigTrivialProducer",
    DTTPGMapBlock,
    DTTPGParametersBlock,
    TracoLutsFromDB = cms.bool(False),
    UseBtiAcceptParam = cms.bool(False)
)



