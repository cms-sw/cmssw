import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_HI_cfi import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1caloparProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TCaloParamsRcd'),
                 tag = cms.string("L1TCaloParamsPrototype_Stage2v0_hlt")
            )
       )
)

L1TCaloParamsOnlineProd = cms.ESProducer("L1TCaloParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(True),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    exclusiveLayer       = cms.uint32(0), # process both layers by default
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)

