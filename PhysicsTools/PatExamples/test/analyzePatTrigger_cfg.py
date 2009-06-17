import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load( "FWCore.MessageService.MessageLogger_cfi" )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:edmPatTrigger.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'rootPatTrigger.root' )
)

process.triggerAnalysis = cms.EDAnalyzer( "PatTriggerAnalyzer",
    trigger      = cms.InputTag( "patTrigger" ),
    triggerEvent = cms.InputTag( "patTriggerEvent" ),
    muons        = cms.InputTag( "selectedLayer1Muons" ),
    muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
    minID = cms.uint32( 81 ),
    maxID = cms.uint32( 102 )
)

process.p = cms.Path(
    process.triggerAnalysis
)
