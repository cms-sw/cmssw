import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTM")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

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
        'file:../../../../../Commissioning2014_Run22913_Cosmics.root'
    )
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:hltonline_2014', '')

process.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32( 813 ),
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    UnpackBxInEvent = cms.int32( 5 ),
    ActiveBoardsMask = cms.uint32( 0xffff )
)

process.hltm = cms.EDAnalyzer('TriggerJSONMonitoring',
    triggerResults = cms.InputTag( 'TriggerResults','','HLT'),
    L1Results = cms.InputTag( "hltGtDigis" )
)

process.p = cms.Path(process.hltGtDigis + process.hltm)
