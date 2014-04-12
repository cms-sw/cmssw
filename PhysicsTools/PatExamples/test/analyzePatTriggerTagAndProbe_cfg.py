import FWCore.ParameterSet.Config as cms

## Declare Process
process = cms.Process( "TEST" )

## Configure MessageLogger
process.load( "FWCore.MessageService.MessageLogger_cfi" )
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )
)

## Declare input
from PhysicsTools.PatExamples.samplesCERN_cff import zjetsTrigger
process.source = cms.Source( "PoolSource",
    fileNames = zjetsTrigger
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

## Define output file
process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'analyzePatTriggerTagAndProbe.root' )
)

## Define tag and probre analyzer
process.tagAndProbeAnalysis = cms.EDAnalyzer( "PatTriggerTagAndProbe",
    triggerEvent = cms.InputTag( "patTriggerEvent" ),
    muons        = cms.InputTag( "cleanPatMuons" ),
    muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' )
)

process.p = cms.Path(
    process.tagAndProbeAnalysis
)
