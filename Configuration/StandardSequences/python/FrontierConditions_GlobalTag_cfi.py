import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

gtDbName = 'frontier://FrontierProd/CMS_CONDITIONS'
print "# CondDB set up to read the Global Tag from : ", gtDbName

GlobalTag = cms.ESSource( "PoolDBESSource",
    CondDBSetup,
    connect = cms.string(gtDbName),
    globaltag = cms.string('UNSPECIFIED'),
    toGet = cms.VPSet( ),   # hook to override or add single payloads
 )
