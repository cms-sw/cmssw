import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
#FRONTIER
RPCCalibPerf = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCStripNoisesRcd'),
        tag = cms.string('Example_tag2')
    )),
    connect = cms.string('frontier://FrontierProd/CMS_COND_20X_RPC'), ##FrontierProd/CMS_COND_20X_RPC"

    authenticationMethod = cms.untracked.uint32(0)
)



