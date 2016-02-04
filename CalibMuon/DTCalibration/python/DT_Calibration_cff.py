import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
#FRONTIER
maps_frontier = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0Fake_18X_Sept15')
    ), 
        cms.PSet(
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('ttrig_18X_July6')
        )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_DT'), ##FrontierDev/CMS_COND_DT"

    authenticationMethod = cms.untracked.uint32(0)
)


