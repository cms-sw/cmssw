import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

print '# Conditions read from new DB via FrontierProd '

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
    globaltag = cms.string('UNSPECIFIED'),
    toGet = cms.VPSet( ),   # hook to override or add single payloads
)
