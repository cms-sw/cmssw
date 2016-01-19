import FWCore.ParameterSet.Config as cms

from CondCore.CondDB.CondDB_cfi import *
CondDB_prod = CondDB.clone( connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS') )

print '# Conditions read from  CMS_CONDITIONS  via FrontierProd '

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDB_prod,
    globaltag = cms.string('UNSPECIFIED'),
    toGet = cms.VPSet( ),   # hook to override or add single payloads
)
