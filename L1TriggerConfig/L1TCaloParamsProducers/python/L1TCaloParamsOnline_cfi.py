import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TCalorimeter.caloStage2Params_cfi import *
from L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_1_cfi import *

L1TCaloParamsOnline = cms.ESProducer("L1TCaloParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)

