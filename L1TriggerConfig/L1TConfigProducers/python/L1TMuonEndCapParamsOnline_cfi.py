import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1emtfparProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('L1TMuonEndCapParamsRcd'),
                tag    = cms.string('L1TMuonEndCapParamsPrototype_Stage2v0_hlt')
            )
       )
)

L1TMuonEndCapParamsOnlineProd = cms.ESProducer("L1TMuonEndCapParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(True),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
