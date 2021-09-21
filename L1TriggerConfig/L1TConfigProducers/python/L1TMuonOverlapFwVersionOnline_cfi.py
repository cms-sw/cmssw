import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonOverlap.fakeOmtfFwVersion_cff import *

#from CondCore.CondDB.CondDB_cfi import CondDB
#CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')
#CondDB.connect = cms.string('oracle://cms_orcon_adg/CMS_CONDITIONS')
#CondDB.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')

#l1omtfparProtodb = cms.ESSource("PoolDBESSource",
#       CondDB,
#       toGet = cms.VPSet(
#            cms.PSet(
#                 record = cms.string('L1TMuonOverlapFwVersionRcd'),
#                 tag    = cms.string("L1TMuonOverlapFwVersionPrototype_Stage2v0_hlt")
#            )
#       )
#)

L1TMuonOverlapFwVersionOnlineProd = cms.ESProducer("L1TMuonOverlapFwVersionOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(False),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
#    onlineDB             = cms.string('oracle://cms_omds_adg/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
