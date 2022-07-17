import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1bmtfparProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TMuonBarrelParamsRcd'),
                 tag    = cms.string("L1TMuonBarrelParamsPrototype_Stage2v0_hlt")
            )
       )
)

L1TMuonBarrelParamsOnlineProd = cms.ESProducer("L1TMuonBarrelParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(True),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
