import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
#    lastRun = cms.untracked.uint32(1),
#    timetype = cms.string('runnumber'),
#    interval = cms.uint32(1),
    firstRun = cms.untracked.uint32(1)
)


#process.TFileService = cms.Service("TFileService",
#                                   fileName = cms.string("siPixelDynamicInefficiency_histo.root")
#                                   )


process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag = autoCond['run2_design']
#In case you of conditions missing, or if you want to test a specific GT
#process.GlobalTag.globaltag = 'PRE_DES72_V6'
print process.GlobalTag.globaltag

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.QualityReader = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('')
    ),
    toGet = cms.VPSet(
		cms.PSet(
			record = cms.string("SiPixelDynamicInefficiencyRcd"),
			tag = cms.string("SiPixelDynamicInefficiency_v1")
		),
	),
    connect = cms.string('sqlite_file:siPixelDynamicInefficiency.db')
)

process.es_prefer_QualityReader = cms.ESPrefer("PoolDBESSource","QualityReader")

process.DynamicInefficiencyReader = cms.EDAnalyzer("SiPixelDynamicInefficiencyReader",
    printDebug = cms.untracked.bool(False)
  )
process.p = cms.Path(process.DynamicInefficiencyReader)
