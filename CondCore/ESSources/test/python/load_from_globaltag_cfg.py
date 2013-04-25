#import FWCore.ParameterSet.Config as cms

#process = cms.Process("TEST")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = cms.string("sqlite_file:FT_53_V21_AN3.db")
#process.CondDBCommon.DBParameters.messageLevel = 0

#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    process.CondDBCommon,
#    globaltag = cms.string('FT_53_V21_AN3::All')
#)

#process.source = cms.Source("EmptyIOVSource",
#    lastValue = cms.uint64(3),
#    timetype = cms.string('runnumber'),
#    firstValue = cms.uint64(1),
#    interval = cms.uint64(1)
#)

#process.get = cms.EDFilter("EventSetupRecordDataGetter",
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('anotherPedestalsRcd'),
#        data = cms.vstring('Pedestals')
#    ), cms.PSet(
#        record = cms.string('PedestalsRcd'),
#        data = cms.vstring('Pedestals/lab3d', 'Pedestals/lab2')
#    )),
#    verbose = cms.untracked.bool(True)
#)

#process.p = cms.Path(process.get)




import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = cms.string("sqlite_file:FT_53_V21_AN3.db")
process.GlobalTag.globaltag = "FT_53_V21_AN3::All"

process.path = cms.Path()
