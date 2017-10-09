import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1emtfparProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                record = cms.string('L1TMuonEndcapParamsRcd'),
                tag = cms.string('L1TMuonEndCapParamsPrototype_Stage2v0_hlt')
            )
       )
)

L1TMuonEndcapParamsOnlineProd = cms.ESProducer("L1TMuonEndcapParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
