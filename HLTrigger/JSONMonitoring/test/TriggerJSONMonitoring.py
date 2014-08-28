import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTM")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.EvFDaqDirector = cms.Service( "EvFDaqDirector",
    buBaseDir = cms.untracked.string( "." ),
    runNumber = cms.untracked.uint32( 0 ),
    baseDir = cms.untracked.string( "." )
)
process.FastMonitoringService = cms.Service( "FastMonitoringService",
    slowName = cms.untracked.string( "slowmoni" ),
    sleepTime = cms.untracked.int32( 1 ),
    fastMonIntervals = cms.untracked.uint32( 2 ),
    fastName = cms.untracked.string( "fastmoni" )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:../../../../../Commissioning2014_Run224380_1000Events.root'
    )
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:hltonline_8E33v2', '')

process.hltm = cms.EDAnalyzer('TriggerJSONMonitoring',
    triggerResults = cms.InputTag( 'TriggerResults','','HLT')
)


process.p = cms.Path(process.hltm)
