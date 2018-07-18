import FWCore.ParameterSet.Config as cms

#from L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff import *

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

l1emtfForestProtodb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TMuonEndCapForestRcd'),
                 tag    = cms.string('L1TMuonEndCapForest_static_Sq_20170613_v7_mc')
            )
       )
)

L1TMuonEndCapForestOnlineProxy = cms.ESProducer("L1TMuonEndCapForestOnlineProxy")
