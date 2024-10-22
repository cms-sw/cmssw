import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TMuon.fakeGmtParams_cff import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1gmtparProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TMuonGlobalParamsRcd'),
                 tag    = cms.string('L1TMuonGlobalParamsPrototype_Stage2v0_hlt')
            )
       )
)

L1TMuonGlobalParamsOnlineProd = cms.ESProducer("L1TMuonGlobalParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(True),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
